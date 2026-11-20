# Tệp, cột dữ liệu và khả năng tái lập

Sáu dòng đầu của bảng kết quả là phần so sánh cuối của sổ đo. Dòng distilled pruned INT8 là lần xếp đã chỉnh, pruning với độ thưa cuối 0.30, rồi INT8. Dòng stacked, default config là lần xếp cùng thứ tự với độ thưa cuối 0.70.

`results/results.csv` ghi đúng tám dòng đang in, từng cột. Độ chính xác, độ giảm, kích thước, p50, p95 và bộ nhớ đỉnh trong báo cáo là đúng các cột `test_acc`, `acc_drop_pp`, `raw_KB`, `gzip_KB`, `size_cut_pct`, `p50_ms`, `p95_ms` và `peak_mem_kB`. Cột `size_cut_pct` là phần cắt gzip. Phần cắt thô không có cột trong CSV. Báo cáo tính `1 - raw_KB / 1690.108`, với 1690.108 là `raw_KB` của float baseline.

File `.tflite` đã đo nằm trong repo này theo bảng dưới. Cột “lúc đo” là đường dẫn trong thư mục thí nghiệm. Chỉ dòng distilled pruned INT8 đổi tên khi đưa vào `models/`.

| Dòng | File trong repo | File lúc đo |
|---|---|---|
| float baseline | `current/baseline_v2.tflite` | `current/baseline_v2.tflite` |
| small from-scratch | `models/v3_student_hard.tflite` | `test/v3_student_hard.tflite` |
| pruned | `models/v3_pruned.tflite` | `test/v3_pruned.tflite` |
| INT8 | `models/v3_int8.tflite` | `test/v3_int8.tflite` |
| distilled student | `models/v3_distill.tflite` | `test/v3_distill.tflite` |
| pruned INT8 | `models/v3_pruned_int8.tflite` | `test/v3_pruned_int8.tflite` |
| distilled pruned INT8 | `models/stack_retuned.tflite` | `test/stack_distill_prune_int8_20261118-124701.tflite` |
| stacked, default config | `models/v3_distill_pruned_int8.tflite` | `test/v3_distill_pruned_int8.tflite` |

Mô hình gốc được nạp từ `locked/baseline.keras`. SHA-256 của file này là `a7d3dc90da3bbf12aaece24ab57a014a216f3cff326d20c0d146434e69dfdb7b`. Bản sao khóa `locked/baseline_v2_locked.keras` có cùng mã băm. File này không bị ghi trong lần đo.

Mẫu số gzip là gzip của file float baseline trong bảng trên, bằng 1,567,603 byte, tức 1567.603 KB ở cột `gzip_KB`. Mẫu số thô là 1690.108 KB ở cột `raw_KB` của cùng dòng.

Các số ở mục phân tích độ nhạy không thuộc tám dòng trên. Chúng nằm ở `results/sweep.csv`. Mỗi dòng của file đó đổi một nút và giữ các nút còn lại ở giá trị đã chốt. Lần quét không đo p50, p95 hay bộ nhớ đỉnh.

Báo cáo được dịch bằng XeTeX với font Liberation, vì T5 Latin Modern trên máy này không có glyph tiếng Việt.
