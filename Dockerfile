# Sử dụng image Python chuẩn
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy file requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn và thư mục models vào container
COPY . .

# Mở port 8000
EXPOSE 8000

# Lệnh khởi chạy server (Sửa app:app thành main:app)
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]