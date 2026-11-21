# SmartShop Product Recommendation Engine

Hệ thống gợi ý sản phẩm thương mại điện tử kết hợp lọc cộng tác, lọc theo nội dung, mô hình lai (Hybrid) và mô hình đa phương thức (Ảnh + Văn bản + Đồ thị + Embedding người dùng/sản phẩm) trên nền Flask.

## Tính năng chính
- **Collaborative Filtering**: Gợi ý dựa trên lịch sử mua sắm của nhóm người dùng có hành vi tương đồng.
- **Content-Based Filtering**: Gợi ý sản phẩm cùng danh mục/thuộc tính dựa trên lịch sử xem sản phẩm.
- **Hybrid Recommendation**: Tổng hợp kết quả từ CF và Content-Based, tự động chuyển về nhóm sản phẩm phổ biến khi thiếu dữ liệu cá nhân hóa.
- **Multi-Modal Deep Learning (`MultiModalModel`)**: Kết hợp embedding người dùng/sản phẩm, đặc trưng ảnh (ResNet50 + góc nhìn), đặc trưng mô tả văn bản (SentenceTransformer) và cấu trúc đồ thị (`GCNConv`).
- **Flask Web App**: Giao diện web hiển thị đầu vào và danh sách sản phẩm gợi ý.

## Cấu trúc thư mục
- `app.py`: Khởi tạo Flask server và điều hướng các route.
- `model.py`: Chứa các hàm gợi ý (`collaborative_filtering`, `content_based_filtering`, `hybrid_recommendation`) và `MultiModalModel`.
- `run_model.py` & `test_model.py`: Chạy pipeline và kiểm thử logic mô hình.
- `templates/` & `static/`: Giao diện HTML và CSS (`index.html`, `recommendations.html`, `styles.css`).

## Cài đặt & Chạy ứng dụng
```bash
pip install -r requirements.txt
python app.py