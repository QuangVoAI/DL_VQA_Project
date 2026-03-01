## VQA FastAPI Service

Thư mục này chứa API đơn giản để deploy 4 mô hình VQA (M1..M4) bằng FastAPI.

### 1. Chuẩn bị artifact từ notebook

Trong `VQA.ipynb`, chạy **cell cuối cùng** (cell tạo `vqa_deploy_all_models.pth`) sau khi:
- Đã train xong các mô hình và có checkpoint `best_M*_*.pth` trong thư mục `checkpoints/`
- Đã có sẵn `question_vocab` và `answer_vocab` trong notebook

File được tạo:

- `vqa_deploy_all_models.pth`

### 2. Cài đặt phụ thuộc

Từ thư mục gốc project:

```bash
pip install -r requirements.txt
```

### 3. Chạy server FastAPI

Từ thư mục gốc project:

```bash
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Gọi API

- Endpoint: `POST /v1/predict`
- Form-data:
  - `image`: file ảnh (JPEG/PNG)
  - JSON body:
    - `question`: câu hỏi VQA
    - `model_name` (tùy chọn): một trong
      - `M1_Scratch_NoAttn`
      - `M2_Scratch_Attn`
      - `M3_Pretrained_NoAttn`
      - `M4_Pretrained_Attn`
    - Nếu bỏ trống `model_name`, API sẽ trả lời bằng tất cả các model.

Ví dụ với `curl`:

```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -F "image=@phone.jpg" \
  -H "Content-Type: multipart/form-data" \
  -d '{"question": "What is the woman doing?", "model_name": "M4_Pretrained_Attn"}'
```

### 5. Health check

- Endpoint: `GET /health`
- Trả về số lượng model đã nạp.

