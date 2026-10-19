# Vòng Sơ loại

## Thành phần

- `model.py`, `predict.py`: phân đoạn mặt đường.
- `road_control.py`: xác định tâm đường và điều khiển bám đường.
- `ucr_main.py`: kết nối API và chạy vòng điều khiển.
- `runtime.env`: cấu hình vận hành.

## Chạy

Chuẩn bị môi trường BTC có `ucr_lib` và các phụ thuộc trong `../requirements.txt`. Đặt các file của thư mục này cùng trọng số `road_selected.pth` vào `/workspace` của container BTC.

```bash
cd /workspace
set -a
. ./runtime.env
set +a
python -u ucr_main.py
```

Mở simulator đúng vòng ở port **11000**, khóa **60 FPS**. Khi chương trình sẵn sàng, bấm **START** và chọn **AV Mode**. Dừng chương trình bằng **Ctrl+C**.
