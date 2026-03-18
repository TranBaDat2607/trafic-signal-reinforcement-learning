# Hướng Dẫn Chạy Dự Án Traffic Signal Reinforcement Learning

> **Mô tả:** Dự án này thiết kế mô hình điều khiển đèn tín hiệu giao thông ngã tư bằng thuật toán Học Tăng Cường (Reinforcement Learning - DQN) kết hợp phần mềm mô phỏng SUMO. Dưới đây là quy trình chạy chuẩn từ lấy mẫu đối chiếu, huấn luyện mô hình đến vẽ biểu đồ so sánh.

---

## 1. Cài đặt môi trường (Chỉ cần chạy 1 lần đầu)
Để cài đặt các phụ thuộc cần thiết (PyTorch, SUMO tools, v.v.), hãy đứng ở thư mục gốc của dự án và chạy dòng lệnh sau (hiện tại dự án đã được tôi chủ động cài đặt trên máy bạn):
```bash
pip install -e .
```

---

## 2. Quy trình chạy (Pipeline)

### Bước 1: Chạy mô phỏng Baseline (Đèn tín hiệu mặc định tĩnh)
Mô phỏng ngã tư hoạt động bằng phương thức đèn xoay vòng cố định (Fixed-time phase). Kết quả này làm cơ sở đánh giá độ hiệu quả của mô hình AI về sau.

```bash
python run_baseline.py --config settings/training_settings.yaml --out baseline_results
```
- **File xuất ra:** Sẽ lưu trữ ở folder `baseline_results/` bao gồm file thiết lập tổng và các đo lường `.txt`.

### Bước 2: Tiến hành Huấn Luyện mô hình Reinforcement Learning (DQN)
Agent AI bắt đầu học cách bật/tắt đèn thông qua tương tác môi trường (Environment - State - Reward).

```bash
python src/train.py --settings settings/training_settings.yaml --out model/run-01
```
- **Mẹo tối ưu:** Với các thiết lập trong `settings/training_settings.yaml` hiện tại (`total_episodes: 10`), AI mới chỉ ở bước trải nghiệm ngẫu nhiên. Để đạt kết quả vượt trội hơn trạng thái tĩnh, hãy mở file config này lên và đổi `total_episodes` thành con số lớn hơn (ví dụ: `200` trở lên) để agent học được các trường hợp tối ưu nhất.
- **File xuất ra:** Sẽ sinh ra thư mục `model/run-01/` chứa các tham số được AI sinh ra và mô hình PyTorch đã train `best_model.pt`.

### Bước 3: Đối chiếu, vẽ biểu đồ so sánh
Sau khi có dữ liệu từ mô hình tĩnh và AI, chúng ta chạy lệnh ghép các dữ liệu trích xuất vào dạng hình ảnh hoặc Text console để báo cáo hiệu suất thu được.

```bash
python compare_results.py --baseline baseline_results --rl model/run-01 --out comparison_output
```
- **File xuất ra:** Hình ảnh `comparison.png` báo cáo biểu đồ tại thư mục `comparison_output/`. Quan sát các đường đồ thị Queue (số xe đang chờ) và Delay (Thời gian kẹt xe) càng thấp xuống qua các Episodes là tín hiệu tốt.

---

## 3. Khởi chạy giao diện trực quan SUMO (GUI)

Mặc định các quá trình huấn luyện bên trên chạy ngầm (headless) cho tốc độ tính toán nhanh nhất. Tuy nhiên, nếu bạn muốn quan sát xe chạy hoạt họa trực tiếp:

1. **Lựa chọn 1:** Mở file `settings/training_settings.yaml` bằng text editor, tìm tới tham số cấu hình:
   ```yaml
   gui: true
   ```
2. **Lựa chọn 2** (Áp dụng cho baseline): Thêm thẻ `--gui` phía sau lệnh bash:
   ```bash
   python run_baseline.py --config settings/training_settings.yaml --gui
   ```

> **Giám sát Dự án - Best Practices:** Luôn lưu kết quả xuất ra ở các lần test với tên folder tăng dần (ví dụ `model/run-02`, `model/run-03`) để đối chiếu lại với nhau khi bạn tiến hành thay đổi siêu tham số (Hyperparameters).
