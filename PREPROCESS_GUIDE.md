# Hướng dẫn sử dụng preprocess.py

Script này giúp tạo một tập con (subset) của VQA v2.0 dataset với số lượng ảnh giới hạn, tiện cho việc thử nghiệm nhanh.

## Cách sử dụng

### 1. Tải VQA v2.0 dataset

Trước tiên, tải các file JSON của VQA v2.0:
- Questions: `v2_OpenEnded_mscoco_train2014_questions.json`
- Annotations: `v2_mscoco_train2014_annotations.json`

Đặt chúng vào thư mục `data/`

### 2. Chạy script để tạo subset

**Cách đơn giản nhất (sử dụng mặc định 5000 ảnh):**
```bash
python preprocess.py
```

**Tùy chỉnh số lượng ảnh:**
```bash
python preprocess.py --num_images 5000
```

**Chỉ định đường dẫn file cụ thể:**
```bash
python preprocess.py \
    --questions data/v2_OpenEnded_mscoco_train2014_questions.json \
    --annotations data/v2_mscoco_train2014_annotations.json \
    --output_questions data/subset_questions.json \
    --output_annotations data/subset_annotations.json \
    --num_images 5000
```

### 3. Kết quả

Script sẽ tạo 2 file mới:
- `data/subset_questions.json` - Chứa câu hỏi cho 5000 ảnh đầu tiên
- `data/subset_annotations.json` - Chứa câu trả lời tương ứng

### 4. Sử dụng với VQADataset

Sau khi có file subset, bạn có thể sử dụng với class `VQADataset`:

```python
from src.dataset import VQADataset, Vocabulary
from torchvision import transforms

# Tạo transforms cho ảnh
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Tạo dataset với subset
dataset = VQADataset(
    image_dir='data/train2014',
    questions_file='data/subset_questions.json',
    annotations_file='data/subset_annotations.json',
    vocab=vocab,  # Vocabulary object của bạn
    transform=transform,
    split='train'
)

print(f"Dataset size: {len(dataset)}")
```

## Tham số

- `--questions`: Đường dẫn file JSON câu hỏi gốc
- `--annotations`: Đường dẫn file JSON annotations gốc
- `--output_questions`: Đường dẫn file JSON câu hỏi subset (output)
- `--output_annotations`: Đường dẫn file JSON annotations subset (output)
- `--num_images`: Số lượng ảnh muốn lấy (mặc định: 5000)

## Ghi chú

- Script sẽ lấy N ảnh **đầu tiên** dựa trên thứ tự xuất hiện trong file JSON
- Tất cả câu hỏi và annotations liên quan đến những ảnh đó sẽ được giữ lại
- Format JSON output giống với format gốc của VQA v2.0
- File subset có thể được sử dụng trực tiếp với `VQADataset` class

## Thống kê ví dụ

Với 5000 ảnh đầu tiên:
- ~25,000 - 30,000 questions (mỗi ảnh có ~5 câu hỏi)
- ~250,000 - 300,000 annotations (mỗi câu hỏi có ~10 câu trả lời)
