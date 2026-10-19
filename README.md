# UIT CAR RACING 2026 — Bảng Chuyên nghiệp

Mã nguồn đội **DBLETK** tại vòng **Sơ loại** và **Khởi động**.

## Thành tích

- **Top 6/67 đội** vòng Sơ loại.
- **Top 6/16 đội** vòng Khởi động.
- **Top 6 chung cuộc**.

## Mã nguồn

- [qualifying/](qualifying/): nhận diện mặt đường và điều khiển vòng Sơ loại.
- [warmup/](warmup/): bám đường, nhận diện biển báo/vật cản, xử lý giao lộ, fountain và phục hồi vòng Khởi động.

**Công nghệ:** Python, PyTorch, OpenCV, LR-ASPP MobileNetV3-Large và YOLO11s.

## Sử dụng

Model được kèm theo từng thư mục. Cần simulator, thư viện `ucr_lib` và môi trường Docker của BTC.

Hướng dẫn chạy: [Sơ loại](qualifying/README.md) · [Khởi động](warmup/README.md).
