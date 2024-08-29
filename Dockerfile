# Sử dụng image base từ Python
FROM python:3.11-slim

# Cài đặt các công cụ xây dựng và thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY . /app/

# Cài đặt các thư viện Python cần thiết
RUN pip install --upgrade pip && \
    pip install flask face_recognition joblib opencv-python-headless

# Mở cổng 5000
EXPOSE 5000

# Chạy ứng dụng
CMD ["python", "detect_face.py"]
