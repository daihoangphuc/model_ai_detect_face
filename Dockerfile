# Chọn image base từ Python
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY detect_face.py /app/
COPY trani_model.py /app/
COPY knn_model.pkl /app/
COPY training_face /app/training_face

# Cài đặt các thư viện cần thiết
RUN pip install flask face_recognition joblib opencv-python-headless

# Mở cổng 5000
EXPOSE 5000

# Chạy ứng dụng
CMD ["python", "detect_face.py"]
