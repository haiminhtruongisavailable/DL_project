"""DL-10 methods copied from dl10_v3.ipynb.

Same split, seed, pruning schedule, distillation loss, INT8 recipe, and
measurement harness. Files are written only under ui/runs/.
"""

import csv
import gzip
import hashlib
import math
import os
import subprocess
import sys
import time

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tf_keras
import tensorflow_model_optimization as tfmot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "ui", "runs")
LOCKED = os.path.join(ROOT, "locked/baseline.keras")
LOCKED_COPY = os.path.join(ROOT, "locked/baseline_v2_locked.keras")
BASELINE_TFLITE = os.path.join(ROOT, "current/baseline_v2.tflite")
EXPECTED_SHA = "a7d3dc90da3bbf12aaece24ab57a014a216f3cff326d20c0d146434e69dfdb7b"
REF_ACC = 0.9158
SAMPLE_NPY = "/tmp/dl10_latency_sample.npy"
OUTPUT_ROOT = RUNS
STAGE_DIR = None
REPORT = None

DISTILL_CFG = {
    "temperature": 4.0,
    "alpha": 0.5,
    "filters": 8,
    "dense_units": 128,
    "epochs": 10,
    "learning_rate": 0.001,
}
PRUNE_CFG = {
    "sparsity": 0.70,
    "epochs": 4,
    "schedule": "PolynomialDecay",
    "layers": "all",
    "power": 3,
    "frequency": 100,
    "learning_rate": 1e-4,
}
INT8_CFG = {
    "calibration_count": 300,
    "mode": "full-int8",
    "per_channel": True,
}

ROWS = []
BASE_GZIP = None
FLOOR_KB = None
_SPLIT = None

SO_DIR = os.path.join(os.path.dirname(tf.__file__), "lite", "python")
VM_CHILD = r"""
import os, sys
sys.path.insert(0, os.environ["SO_DIR"])
import _pywrap_tensorflow_interpreter_wrapper as wrap
import numpy as np
model = os.environ.get("MODEL") or ""
if model:
    interp = wrap.CreateWrapperFromFile(model, 1, [], [], False, False, 1, False)
    if not interp:
        raise SystemExit("create failed")
    interp.AllocateTensors()
    in_idx = int(list(interp.InputIndices())[0])
    out_idx = int(list(interp.OutputIndices())[0])
    shape = [int(v) for v in interp.TensorSize(in_idx, 0)]
    if shape != [1, 28, 28, 1]:
        interp.ResizeInputTensor(in_idx, np.array([1, 28, 28, 1], np.int32), True)
        interp.AllocateTensors()
    image = np.load(os.environ["NPY"]).astype(np.float32)[None, ...]
    quant = interp.TensorQuantization(in_idx, 0)
    dtype = np.dtype(interp.TensorType(in_idx, 0))
    if dtype == np.int8:
        scale, zero = float(quant[0]), int(quant[1])
        feed = np.clip(np.round(image / scale + zero), -128, 127).astype(np.int8)
    elif dtype == np.float32:
        feed = image
    else:
        raise SystemExit("unexpected input dtype " + dtype.name)
    for _ in range(200):
        interp.SetTensor(in_idx, feed)
        interp.Invoke()
        interp.GetTensor(out_idx, 0)
if "tensorflow" in sys.modules or any(name.startswith("tensorflow.") for name in sys.modules):
    raise SystemExit("tensorflow was imported")
vm = None
with open("/proc/self/status") as status:
    for line in status:
        if line.startswith("VmHWM:"):
            vm = int(line.split()[1])
print(vm)
"""


def gzip_size(payload):
    return len(gzip.compress(payload, compresslevel=9, mtime=0))


def load_split():
    global _SPLIT
    if _SPLIT is not None:
        return _SPLIT
    tf.random.set_seed(42)
    np.random.seed(42)
    (x_train_full, y_train_full), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_full = (x_train_full / 255.0).astype("float32")[..., None]
    x_te = (x_te / 255.0).astype("float32")[..., None]
    x_tr, y_tr = x_train_full[:-6000], y_train_full[:-6000]
    x_val, y_val = x_train_full[-6000:], y_train_full[-6000:]
    assert len(x_tr) == 54000 and len(x_val) == 6000 and len(x_te) == 10000
    if not os.path.exists(SAMPLE_NPY):
        np.save(SAMPLE_NPY, x_te[0])
    _SPLIT = (x_tr, y_tr, x_val, y_val, x_te, y_te)
    return _SPLIT


def base_gzip():
    global BASE_GZIP
    if BASE_GZIP is None:
        with open(BASELINE_TFLITE, "rb") as handle:
            BASE_GZIP = gzip_size(handle.read())
    return BASE_GZIP


def runs_path(path):
    path = os.path.abspath(path)
    root = os.path.abspath(RUNS)
    if path != root and not path.startswith(root + os.sep):
        raise RuntimeError("refusing to write outside ui/runs: " + path)
    return path


def _writable(path):
    path = os.path.abspath(path)
    roots = [os.path.abspath(RUNS), os.path.abspath(os.path.join(ROOT, "test"))]
    for root in roots:
        if path == root or path.startswith(root + os.sep):
            return path
    raise RuntimeError("refusing to write outside ui/runs or test: " + path)


def save_new(path, payload):
    path = _writable(path)
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite " + path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def new_run_dir():
    os.makedirs(RUNS, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RUNS, stamp)
    suffix = 0
    while os.path.exists(path):
        suffix += 1
        path = os.path.join(RUNS, f"{stamp}-{suffix}")
    os.makedirs(path)
    return path


def make_cnn(width):
    return tf_keras.Sequential([
        tf_keras.layers.Input((28, 28, 1)),
        tf_keras.layers.Conv2D(width, 3, activation="relu", padding="same"),
        tf_keras.layers.MaxPooling2D(),
        tf_keras.layers.Conv2D(2 * width, 3, activation="relu", padding="same"),
        tf_keras.layers.MaxPooling2D(),
        tf_keras.layers.Flatten(),
        tf_keras.layers.Dense(128, activation="relu"),
        tf_keras.layers.Dropout(0.3),
        tf_keras.layers.Dense(10, activation="softmax"),
    ])


def first_kernel(model):
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            return layer.name, weights[0]
    raise RuntimeError("no kernel")


def make_student(width, dense_units):
    dense_units = int(dense_units)
    if dense_units == 128:
        return make_cnn(int(width))
    return tf_keras.Sequential([
        tf_keras.layers.Input((28, 28, 1)),
        tf_keras.layers.Conv2D(int(width), 3, activation="relu", padding="same"),
        tf_keras.layers.MaxPooling2D(),
        tf_keras.layers.Conv2D(2 * int(width), 3, activation="relu", padding="same"),
        tf_keras.layers.MaxPooling2D(),
        tf_keras.layers.Flatten(),
        tf_keras.layers.Dense(dense_units, activation="relu"),
        tf_keras.layers.Dropout(0.3),
        tf_keras.layers.Dense(10, activation="softmax"),
    ])


def reload_baseline():
    model = tf_keras.models.load_model(LOCKED)
    digest = hashlib.sha256(open(LOCKED, "rb").read()).hexdigest()
    if digest != EXPECTED_SHA:
        raise SystemExit("locked baseline hash mismatch")
    return model


def load_start(path):
    path = os.path.abspath(path)
    if os.path.realpath(path) == os.path.realpath(LOCKED):
        return reload_baseline()
    return tf_keras.models.load_model(path, compile=False)


def quantize_input(images, detail):
    if detail["dtype"] == np.float32:
        return images.astype(np.float32)
    scale, zero = detail["quantization"]
    values = np.round(images / scale + zero)
    info = np.iinfo(detail["dtype"])
    return np.clip(values, info.min, info.max).astype(detail["dtype"])


def dequantize_output(values, detail):
    if detail["dtype"] == np.float32:
        return values
    scale, zero = detail["quantization"]
    return scale * (values.astype(np.float32) - zero)


def make_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path, num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def tflite_accuracy(path, images, labels):
    interpreter = make_interpreter(path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    batch = 256
    try:
        interpreter.resize_tensor_input(input_detail["index"], [batch, 28, 28, 1], strict=True)
        interpreter.allocate_tensors()
    except (ValueError, RuntimeError):
        batch = 1
    correct = 0
    for start in range(0, len(labels), batch):
        chunk = images[start:start + batch]
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        if len(chunk) != batch:
            interpreter.resize_tensor_input(input_detail["index"], [len(chunk), 28, 28, 1], strict=True)
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_detail["index"], quantize_input(chunk, input_detail))
        interpreter.invoke()
        logits = dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        correct += int((logits.argmax(axis=-1) == labels[start:start + len(chunk)]).sum())
    return correct / len(labels)


def latency_medians(path):
    _, _, _, _, x_te, _ = load_split()
    interpreter = make_interpreter(path)
    detail = interpreter.get_input_details()[0]
    if list(detail["shape"]) != [1, 28, 28, 1]:
        interpreter.resize_tensor_input(detail["index"], [1, 28, 28, 1], strict=True)
        interpreter.allocate_tensors()
    sample = x_te[0][None, ...]
    p50, p95 = [], []
    for _ in range(3):
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        feed = quantize_input(sample, input_detail)
        for _ in range(20):
            interpreter.set_tensor(input_detail["index"], feed)
            interpreter.invoke()
            interpreter.get_tensor(output_detail["index"])
        samples = []
        for _ in range(200):
            start = time.perf_counter()
            interpreter.set_tensor(input_detail["index"], feed)
            interpreter.invoke()
            interpreter.get_tensor(output_detail["index"])
            samples.append((time.perf_counter() - start) * 1000)
        p50.append(float(np.percentile(samples, 50)))
        p95.append(float(np.percentile(samples, 95)))
    p50.sort()
    p95.sort()
    return p50[1], p95[1]


def _vmhwm_once(model_path):
    env = os.environ.copy()
    env["SO_DIR"] = SO_DIR
    env["MODEL"] = model_path
    env["NPY"] = SAMPLE_NPY
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    completed = subprocess.run(
        [sys.executable, "-c", VM_CHILD],
        env=env, check=True, capture_output=True, text=True,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def peak_delta(model_path):
    global FLOOR_KB
    if FLOOR_KB is None:
        empty = sorted(_vmhwm_once("") for _ in range(3))
        FLOOR_KB = empty[1]
        print("empty floor kB", FLOOR_KB, empty)
    readings = sorted(_vmhwm_once(model_path) for _ in range(3))
    return readings[1], readings[1] - FLOOR_KB


def bar_flags(row):
    return row["loss_pp"] <= 2.0, row["size_cut"] >= 50.0


def verdict(row):
    if row.get("reference"):
        return "reference"
    acc_ok, size_ok = bar_flags(row)
    return "PASS" if acc_ok and size_ok else "FAIL"


def measure(path, label):
    _, _, _, _, x_te, y_te = load_split()
    payload = open(path, "rb").read()
    acc = tflite_accuracy(path, x_te, y_te)
    raw = len(payload)
    gzip_n = gzip_size(payload)
    p50, p95 = latency_medians(path)
    vm_median, vm_delta = peak_delta(path)
    row = {
        "name": label,
        "file": os.path.basename(path),
        "path": path,
        "test_acc": acc,
        "loss_pp": (REF_ACC - acc) * 100,
        "raw_kb": raw / 1000,
        "gzip_kb": gzip_n / 1000,
        "raw_mb": raw / 1e6,
        "gzip_mb": gzip_n / 1e6,
        "size_cut": (base_gzip() - gzip_n) / base_gzip() * 100,
        "p50_ms": p50,
        "p95_ms": p95,
        "vmhwm_kb": vm_median,
        "vm_delta_kb": vm_delta,
        "reference": False,
    }
    acc_ok, size_ok = bar_flags(row)
    row["acc_bar"] = "PASS" if acc_ok else "FAIL"
    row["size_bar"] = "PASS" if size_ok else "FAIL"
    row["verdict"] = verdict(row)
    return row


def measure_saved(name, path, reference=False):
    row = measure(path, name)
    if reference:
        row["reference"] = True
        row["size_cut"] = 0.0
        row["acc_bar"] = "reference"
        row["size_bar"] = "reference"
        row["verdict"] = "reference"
    ROWS.append(row)
    print_row(row)
    return row


def save_plot(fig, name):
    folder = os.path.join(ROOT, "test", "plots")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if os.path.dirname(path) != folder:
        raise RuntimeError("refusing a plot path outside test/plots: " + path)
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite " + path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def append_playground(row, order):
    path = os.path.join(ROOT, "test", "playground_runs.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "label", "order", "test_acc", "loss_pp", "raw_mb", "gzip_mb",
        "size_cut", "p50_ms", "p95_ms", "vm_delta_kb", "verdict", "path",
    ]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "label": row["name"],
            "order": "+".join(order),
            "test_acc": f"{row['test_acc']:.4f}",
            "loss_pp": f"{row['loss_pp']:.2f}",
            "raw_mb": f"{row['raw_mb']:.6f}",
            "gzip_mb": f"{row['gzip_mb']:.6f}",
            "size_cut": f"{row['size_cut']:.2f}",
            "p50_ms": f"{row['p50_ms']:.3f}",
            "p95_ms": f"{row['p95_ms']:.3f}",
            "vm_delta_kb": int(row["vm_delta_kb"]),
            "verdict": row["verdict"],
            "path": row["path"],
        })
    print("playground csv", path)
    return path


def print_row(row):
    print(
        f"{row['name']}: test {row['test_acc']:.4f}, loss {row['loss_pp']:.2f} pp, "
        f"raw {row['raw_mb']:.6f} MB, gzip {row['gzip_mb']:.6f} MB, "
        f"size cut {row['size_cut']:.2f}%, p50 {row['p50_ms']:.3f} ms, p95 {row['p95_ms']:.3f} ms, "
        f"VmHWM-floor {row['vm_delta_kb']} kB"
    )
    if row.get("reference"):
        print("accuracy bar (<= 2 pp): not applied")
        print("size bar (>= 50% gzip cut): not applied")
        print("row: reference")
        return
    print("accuracy bar (<= 2 pp):", row["acc_bar"])
    print("size bar (>= 50% gzip cut):", row["size_bar"])


def convert_float(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_int8(model, calibration_count=300, mode="full-int8", per_channel=True):
    x_tr, _, _, _, _, _ = load_split()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    mode = mode or "full-int8"
    if mode == "full-int8":
        yielded = {"n": 0}

        def representative():
            for index in range(int(calibration_count)):
                yielded["n"] += 1
                yield [np.ascontiguousarray(x_tr[index:index + 1])]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        if not per_channel:
            converter._experimental_disable_per_channel = True
        payload = converter.convert()
        print("representative_dataset yields", yielded["n"], "requested", int(calibration_count), flush=True)
        return payload
    elif mode == "dynamic-range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if not per_channel:
            converter._experimental_disable_per_channel = True
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        raise ValueError("unknown INT8 mode " + str(mode))
    return converter.convert()


def _prune_schedule(final_sparsity, epochs, kind, frequency):
    steps = math.ceil(54000 / 128) * int(epochs)
    if kind == "ConstantSparsity":
        return tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=float(final_sparsity),
            begin_step=0,
            frequency=frequency,
        )
    return tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=float(final_sparsity),
        begin_step=0,
        end_step=steps,
        power=3,
        frequency=frequency,
    )


def prune(model, cfg):
    x_tr, y_tr, x_val, y_val, _, _ = load_split()
    epochs = int(cfg.get("epochs", 4))
    sparsity = float(cfg.get("sparsity", 0.70))
    schedule_name = cfg.get("schedule", "PolynomialDecay")
    layers = cfg.get("layers", "all")
    from_distill = bool(cfg.get("from_distill"))
    frequency = 1 if from_distill else int(cfg.get("frequency", 100))
    learning_rate = float(cfg.get("learning_rate", 1e-4))
    schedule = _prune_schedule(sparsity, epochs, schedule_name, frequency)
    if layers == "all":
        pruned = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=schedule)
    else:
        want = tf_keras.layers.Conv2D if layers == "conv" else tf_keras.layers.Dense

        def clone(layer):
            if isinstance(layer, want):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule=schedule)
            return layer

        pruned = tf_keras.models.clone_model(model, clone_function=clone)
    if from_distill:
        print("prune loss: distillation")
        print("prune frequency", frequency)
        teacher = reload_baseline()
        probs = teacher.predict(x_tr, batch_size=256, verbose=0)
        targets = distill_targets(probs, y_tr)
        temperature = float(cfg.get("temperature", DISTILL_CFG["temperature"]))
        alpha = float(cfg.get("alpha", DISTILL_CFG["alpha"]))
        pruned.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate),
            loss=make_distill_loss(temperature, alpha),
        )
        pruned.fit(
            x_tr, targets,
            epochs=epochs,
            batch_size=128,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        )
    else:
        print("prune loss: hard labels")
        print("prune frequency", frequency)
        pruned.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        pruned.fit(
            x_tr, y_tr,
            epochs=epochs,
            batch_size=128,
            validation_data=(x_val, y_val),
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        )
    return tfmot.sparsity.keras.strip_pruning(pruned)


def distill_targets(teacher_probs, labels):
    return np.concatenate([teacher_probs.astype("float32"), labels.astype("float32")[:, None]], axis=1)


def make_distill_loss(temperature=4.0, alpha=0.5):
    def loss(y_true, y_pred):
        teacher = tf.clip_by_value(y_true[:, :10], 1e-8, 1.0)
        hard = tf.cast(y_true[:, 10], tf.int32)
        student = tf.clip_by_value(y_pred, 1e-8, 1.0)
        inv_t = 1.0 / temperature
        soft_teacher = tf.pow(teacher, inv_t)
        soft_student = tf.pow(student, inv_t)
        soft_teacher = soft_teacher / tf.reduce_sum(soft_teacher, axis=-1, keepdims=True)
        soft_student = soft_student / tf.reduce_sum(soft_student, axis=-1, keepdims=True)
        kl = tf.reduce_sum(soft_teacher * (tf.math.log(soft_teacher) - tf.math.log(soft_student)), axis=-1)
        hard_loss = tf_keras.losses.sparse_categorical_crossentropy(hard, y_pred)
        return (1.0 - alpha) * hard_loss + alpha * (temperature ** 2) * kl
    return loss


def distill(model, cfg):
    x_tr, y_tr, x_val, y_val, _, _ = load_split()
    temperature = float(cfg.get("temperature", 4.0))
    alpha = float(cfg.get("alpha", 0.5))
    filters = int(cfg.get("filters", 8))
    dense_units = int(cfg.get("dense_units", 128))
    epochs = int(cfg.get("epochs", 10))
    learning_rate = float(cfg.get("learning_rate", 0.001))
    teacher_train = model.predict(x_tr, batch_size=256, verbose=0)
    targets = distill_targets(teacher_train, y_tr)
    seed = int(cfg.get("seed", 42))
    tf.random.set_seed(seed)
    np.random.seed(seed)
    student = make_student(filters, dense_units)
    if learning_rate == 0.001:
        optimizer = "adam"
    else:
        optimizer = tf_keras.optimizers.Adam(learning_rate=learning_rate)
    student.compile(optimizer=optimizer, loss=make_distill_loss(temperature, alpha))
    best = {"acc": -1.0, "epoch": None, "weights": None}
    for epoch in range(epochs):
        student.fit(x_tr, targets, epochs=1, batch_size=128, shuffle=True, verbose=2)
        acc = float((student.predict(x_val, batch_size=256, verbose=0).argmax(1) == y_val).mean())
        print("val acc", epoch + 1, acc)
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": epoch + 1, "weights": [item.copy() for item in student.get_weights()]}
    student.set_weights(best["weights"])
    print("kept epoch", best["epoch"], "val", best["acc"])
    return student


def int8(model, cfg):
    payload = convert_int8(
        model,
        calibration_count=int(cfg.get("calibration_count", 300)),
        mode=cfg.get("mode", "full-int8"),
        per_channel=bool(cfg.get("per_channel", True)),
    )
    directory = cfg.get("out_dir") or STAGE_DIR
    if not directory:
        raise RuntimeError("int8 has no output directory")
    return save_new(os.path.join(directory, "model.tflite"), payload)


def _is_distilled_student(path):
    name = os.path.basename(path)
    return "distill" in name and "pruned" not in name and not name.startswith("baseline")


STAGES = {"distill": distill, "prune": prune, "int8": int8}


def run_pipeline(start, order, label):
    """Run distill, prune, and int8 in order. Reject a bad order before any fit."""
    order = list(order)
    unknown = [name for name in order if name not in STAGES]
    if unknown:
        raise ValueError("unknown stage: " + ", ".join(str(name) for name in unknown))
    if len(order) != len(set(order)):
        raise ValueError("repeated stage")
    if "int8" in order and order[-1] != "int8":
        raise ValueError("int8 must be last")

    def report(frac, msg):
        fn = REPORT or (lambda frac, msg: print(msg, flush=True))
        fn(frac, msg)

    report(0.0, "order " + " ".join(order))
    tf.random.set_seed(42)
    np.random.seed(42)
    if isinstance(start, str):
        model = load_start(start)
        came_from_distill = _is_distilled_student(start)
    else:
        model = start
        came_from_distill = False
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_") or "run"
    while "__" in slug:
        slug = slug.replace("__", "_")
    run_dir = os.path.join(OUTPUT_ROOT, slug)
    if os.path.exists(run_dir):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(OUTPUT_ROOT, slug + "_" + stamp)
        suffix = 0
        while os.path.exists(run_dir):
            suffix += 1
            run_dir = os.path.join(OUTPUT_ROOT, slug + "_" + stamp + "_" + str(suffix))
    os.makedirs(run_dir)
    global STAGE_DIR
    STAGE_DIR = run_dir
    for index, name in enumerate(order):
        report(index / max(len(order), 1), name)
        if name == "prune":
            cfg = dict(PRUNE_CFG)
            cfg["from_distill"] = came_from_distill
            if came_from_distill:
                cfg["temperature"] = DISTILL_CFG["temperature"]
                cfg["alpha"] = DISTILL_CFG["alpha"]
            model = prune(model, cfg)
            came_from_distill = False
        elif name == "distill":
            model = distill(model, DISTILL_CFG)
            came_from_distill = True
        else:
            path = int8(model, INT8_CFG)
            row = measure(path, label)
            row["run_dir"] = run_dir
            print_row(row)
            report(1.0, "measured " + path)
            return row
    payload = convert_float(model)
    path = save_new(os.path.join(run_dir, "model.tflite"), payload)
    row = measure(path, label)
    row["run_dir"] = run_dir
    print_row(row)
    report(1.0, "measured " + path)
    return row


def measure_baseline_row():
    return measure_saved("Float baseline (current/baseline_v2.tflite)", BASELINE_TFLITE, reference=True)


def card(row):
    if row is None:
        return "No run yet."
    return "\n".join([
        f"**{row['name']}**",
        f"- Test accuracy: {row['test_acc']:.4f}",
        f"- Loss: {row['loss_pp']:.2f} pp vs 0.9158",
        f"- Raw: {row['raw_mb']:.6f} MB",
        f"- Gzip: {row['gzip_mb']:.6f} MB",
        f"- Size cut: {row['size_cut']:.2f}%",
        f"- p50: {row['p50_ms']:.3f} ms",
        f"- p95: {row['p95_ms']:.3f} ms",
        f"- Peak memory: {row['vm_delta_kb']} kB above the empty floor",
        f"- Accuracy bar (<= 2 pp): {row['acc_bar']}",
        f"- Size bar (>= 50% gzip cut): {row['size_bar']}",
        f"- Verdict: {row['verdict']}",
    ])
