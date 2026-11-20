"""Read-only Keras versus float TFLite check.

Logic recreated from the 2026-11-18 one-off stdin run. The saved
max absolute difference was not recomputed when this file was added.
Opens locked/baseline.keras and never saves it.
"""

import hashlib
import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOCKED = os.path.join(ROOT, "locked", "baseline.keras")
TFLITE = os.path.join(ROOT, "current", "baseline_v2.tflite")
EXPECTED = "a7d3dc90da3bbf12aaece24ab57a014a216f3cff326d20c0d146434e69dfdb7b"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    before = sha256(LOCKED)
    print("sha_before", before, "match", before == EXPECTED, flush=True)
    if before != EXPECTED:
        raise SystemExit(2)

    import numpy as np
    import tensorflow as tf
    import tf_keras
    from ui.core import load_split

    _, _, _, _, x_te, y_te = load_split()
    assert len(x_te) == 10000
    model = tf_keras.models.load_model(LOCKED)
    keras_probs = model.predict(x_te, batch_size=256, verbose=0)

    interpreter = tf.lite.Interpreter(model_path=TFLITE, num_threads=1)
    interpreter.allocate_tensors()
    batch = 256
    try:
        interpreter.resize_tensor_input(
            interpreter.get_input_details()[0]["index"],
            [batch, 28, 28, 1],
            strict=True,
        )
        interpreter.allocate_tensors()
    except (ValueError, RuntimeError):
        batch = 1
    tflite_probs = np.empty_like(keras_probs)
    for start in range(0, len(x_te), batch):
        chunk = x_te[start:start + batch]
        interpreter.resize_tensor_input(
            interpreter.get_input_details()[0]["index"],
            [len(chunk), 28, 28, 1],
            strict=True,
        )
        interpreter.allocate_tensors()
        detail_in = interpreter.get_input_details()[0]
        detail_out = interpreter.get_output_details()[0]
        interpreter.set_tensor(detail_in["index"], chunk.astype(np.float32))
        interpreter.invoke()
        tflite_probs[start:start + len(chunk)] = interpreter.get_tensor(detail_out["index"])

    diff = np.abs(keras_probs.astype(np.float64) - tflite_probs.astype(np.float64))
    max_diff = float(diff.max())
    print("n_images", len(x_te), flush=True)
    print("max_abs_prob_diff", repr(max_diff), flush=True)
    print("argmax_agree", float((keras_probs.argmax(1) == tflite_probs.argmax(1)).mean()), flush=True)
    print("keras_acc", float((keras_probs.argmax(1) == y_te).mean()), flush=True)
    print("tflite_acc", float((tflite_probs.argmax(1) == y_te).mean()), flush=True)
    after = sha256(LOCKED)
    print("sha_after", after, "unchanged", after == before, flush=True)
    if after != before:
        raise SystemExit("locked baseline hash changed")


if __name__ == "__main__":
    main()
