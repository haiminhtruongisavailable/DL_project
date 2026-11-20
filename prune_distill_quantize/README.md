# DL-10: nén CNN trên Fashion-MNIST

Bản nộp cuối, ngày 2026-11-21. Báo cáo không còn Phụ lục A. Ánh xạ từng dòng kết quả sang file nằm ở `GUIDE.md`. SHA-256 của các file nộp nằm ở `MANIFEST.txt`.

`notebook/dl10_v3.ipynb` trỏ `ROOT` tới thư mục cha của `notebook/`, tức gốc repo này. Output đã lưu không đổi.

## Thư mục

- `GUIDE.md`: tệp đã đo, cột của `results.csv`, mẫu số kích thước, và mã băm của `locked/baseline.keras`.
- `report/`: báo cáo (`main.tex`), `figures/`, và `DL10_report_final.pdf` (35 trang).
- `notebook/`: `dl10_v3.ipynb`. Mở sổ với thư mục làm việc là `notebook/`. Khi đó `ROOT` là thư mục cha, tức gốc repo này.
- `results/`: `results.csv` (tám dòng đã chốt), `sweep.csv` (quét một nút), `equivalence.json` (lệch Keras–TFLite đã lưu).
- `scripts/`: `check_equivalence.py`. Script chỉ đọc model, không ghi `locked/`.
- `ui/`: `core.py`, hàm `load_split`. `ui/__init__.py` là file rỗng.
- `locked/`: `baseline.keras`, chỉ đọc.
- `current/`: `baseline_v2.tflite`, file float baseline.
- `models/`: file `.tflite` cuối của từng dòng nén. Notebook không đọc thư mục này. Notebook vẫn ghi vào `test/` và từ chối ghi đè nếu file đã có.

## `models/`

Kích thước thô là byte/1000, cùng đơn vị cột `raw_KB`.

| File | Dòng trong bảng | Raw KB |
|---|---|---:|
| `current/baseline_v2.tflite` | float baseline | 1690.108 |
| `models/v3_student_hard.tflite` | small from-scratch | 415.612 |
| `models/v3_pruned.tflite` | pruned | 1690.108 |
| `models/v3_int8.tflite` | INT8 | 431.992 |
| `models/v3_distill.tflite` | distilled student | 415.612 |
| `models/v3_pruned_int8.tflite` | pruned INT8 | 431.992 |
| `models/v3_distill_pruned_int8.tflite` | stacked, default config | 111.424 |
| `models/stack_retuned.tflite` | distilled pruned INT8 | 111.424 |

`models/stack_retuned.tflite` là bản sao của `test/stack_distill_prune_int8_20261118-124701.tflite`. Chỉ tên trong repo được đổi.

## Ba vạch

Một dòng nén PASS khi cả ba điều kiện sau đều đúng. Dòng float baseline là mốc, không bị chấm.

1. Độ giảm accuracy không quá 2 percentage point so với 0.9158.
2. Raw cut ít nhất 50% và gzip cut ít nhất 50%. Raw cut = \(1 - \mathrm{raw\_KB}/1690.108\). Gzip cut lấy từ cột `size_cut_pct`.
3. Một lần, trên mô hình float, trước quantization: sai lệch đầu ra lớn nhất giữa Keras và TFLite không quá \(1\times 10^{-4}\). Giá trị đã lưu là \(1.7583370208740234\times 10^{-6}\), trong `results/equivalence.json`.

## Chạy `scripts/check_equivalence.py`

Từ gốc repo:

```bash
TF_USE_LEGACY_KERAS=1 python scripts/check_equivalence.py
```

Cài gói trong `requirements.txt` trước. Script lấy gốc dự án là thư mục cha của `scripts/`. Nó đọc `locked/baseline.keras`, `current/baseline_v2.tflite`, và gọi `from ui.core import load_split`.

`load_split` lấy 10.000 ảnh test của Fashion-MNIST qua `tf.keras.datasets` (`/255`, `float32`, có channel axis). Script nạp Keras bằng `tf_keras`, batch 256, và chạy `tf.lite.Interpreter` với `num_threads=1`. Giá trị trong `equivalence.json` chưa được đo lại bằng file script này.

Chạy lại notebook tạo file dưới `test/`. Nếu tên đã tồn tại, sổ dừng và không ghi đè. `test/` và `results/raw_measurements.json` nằm trong `.gitignore`.

SHA-256 của `locked/baseline.keras` là `a7d3dc90da3bbf12aaece24ab57a014a216f3cff326d20c0d146434e69dfdb7b`.
