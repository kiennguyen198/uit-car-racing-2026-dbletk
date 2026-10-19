# UIT CAR RACING 2026 — Bảng Chuyên nghiệp

## DBLETK · Top 6/67 đội tại vòng Sơ loại

Mã nguồn của đội **DBLETK** cho hai vòng **Sơ loại** và **Khởi động** thuộc cuộc thi **UIT CAR RACING 2026 — Bảng Chuyên nghiệp**.

## Thành tích

- **Top 6 chung cuộc**, trong cuộc thi có **67 đội tham gia**.


## Cấu trúc

| Thư mục | Nội dung |
| --- | --- |
| [`qualifying/`](qualifying/) | Phân đoạn mặt đường và điều khiển bám đường vòng Sơ loại |
| [`warmup/`](warmup/) | Phân đoạn mặt đường, nhận diện biển báo/vật cản, xử lý giao lộ, fountain và phục hồi vòng Khởi động |

## Công nghệ

Python · PyTorch · Torchvision · OpenCV · LR-ASPP MobileNetV3-Large · YOLO11s

## Sử dụng

Cần simulator, thư viện `ucr_lib` và môi trường Docker do BTC cung cấp. Trọng số model cần thiết được kèm theo từng thư mục vòng thi. Xem hướng dẫn tại [Sơ loại](qualifying/README.md) và [Khởi động](warmup/README.md).
