# Vòng Khởi động

## Thành phần

- `model.py`, `predict.py`: phân đoạn mặt đường.
- `detector.py`: nhận diện biển báo, phương tiện và fountain.
- `road_control.py`, `traffic_control.py`: bám đường, xử lý giao lộ và vật cản.
- `fountain_policy.py`, `fountain_memory.py`: xử lý fountain.
- `stuck_recovery.py`: phục hồi từ trạng thái mất đường hoặc đứng yên.
- `ucr_main.py`, `controller.py`: khởi chạy và điều phối vòng điều khiển.
- `runtime.env`: cấu hình vận hành.

## Chạy

Chuẩn bị môi trường BTC có `ucr_lib`, CUDA và các phụ thuộc trong `../requirements.txt`. Đặt toàn bộ file của thư mục này vào `/workspace`, gồm hai trọng số đã kèm theo:

- `road_selected.pth`.
- `map2_signs_obstacles_fountain_yolo11s_v9_clean.torchscript`.

```bash
cd /workspace
set -a
. ./runtime.env
set +a
python -u ucr_main.py
```

Mở simulator đúng vòng ở port **11000**, khóa **60 FPS**. Khi terminal hiện `SẴN SÀNG`, bấm **START** và chọn **AV Mode**. Dừng chương trình bằng **Ctrl+C**.
