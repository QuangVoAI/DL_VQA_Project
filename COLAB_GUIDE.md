# VQA Training on Google Colab - Hướng dẫn

Các Jupyter Notebooks để train VQA model trên Google Colab.

## Notebooks

### 1. `01_preprocess.ipynb` - Data Preprocessing
Tạo subset từ VQA v2.0 dataset

**Chức năng:**
- Tải VQA v2.0 dataset
- Tạo subset với N ảnh đầu tiên (mặc định 5000)
- Lưu file JSON subset để sử dụng cho training

**Kết quả:**
- `data/subset_questions.json`
- `data/subset_annotations.json`

### 2. `02_train_vqa.ipynb` - VQA Model Training
Train VQA model với ResNet + LSTM

**Chức năng:**
- Định nghĩa Vocabulary, Dataset, Model
- Train model trên subset data
- Lưu checkpoints
- Inference và visualize kết quả

**Kết quả:**
- Model checkpoints trong `checkpoints/`
- Training history plots

## Cách sử dụng trên Google Colab

### Bước 1: Upload notebooks lên Colab
1. Mở Google Colab: https://colab.research.google.com/
2. Upload `01_preprocess.ipynb` và `02_train_vqa.ipynb`

### Bước 2: Chuẩn bị dữ liệu
Chạy `01_preprocess.ipynb`:
1. Mount Google Drive (nếu cần)
2. Download VQA v2.0 dataset hoặc upload sẵn vào Drive
3. Chạy các cells để tạo subset

### Bước 3: Training
Chạy `02_train_vqa.ipynb`:
1. Đảm bảo đã có subset data từ bước 2
2. Chọn GPU runtime: Runtime → Change runtime type → GPU
3. Chạy các cells theo thứ tự
4. Monitor training progress

## Yêu cầu

### Runtime
- **Khuyến nghị:** GPU (T4, P100, hoặc tốt hơn)
- **RAM:** Standard (12GB) hoặc High-RAM (25GB)

### Data
Tải từ: https://visualqa.org/download.html
- Training Questions (v2.0)
- Training Annotations (v2.0)  
- Training Images (COCO 2014)

## Cấu trúc thư mục

```
├── data/
│   ├── train2014/                           # COCO images
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_mscoco_train2014_annotations.json
│   ├── subset_questions.json                # Tạo bởi 01_preprocess.ipynb
│   └── subset_annotations.json              # Tạo bởi 01_preprocess.ipynb
├── checkpoints/
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   └── ...
├── 01_preprocess.ipynb
└── 02_train_vqa.ipynb
```

## Hyperparameters (trong 02_train_vqa.ipynb)

```python
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 32          # Giảm nếu bị OOM
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
TOP_K_ANSWERS = 1000
MAX_QUESTION_LENGTH = 20
```

## Tips

### Nếu bị Out of Memory (OOM):
- Giảm `BATCH_SIZE` xuống 16 hoặc 8
- Sử dụng High-RAM runtime
- Giảm `TOP_K_ANSWERS` xuống 500

### Lưu checkpoints vào Google Drive:
```python
# Thêm vào đầu notebook
from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/vqa_checkpoints'
```

### Theo dõi training với TensorBoard:
```python
# Cài đặt
!pip install tensorboard

# Load extension
%load_ext tensorboard

# Start tensorboard
%tensorboard --logdir checkpoints/
```

## Model Architecture

### EncoderCNN
- Backbone: ResNet-50 (pretrained on ImageNet)
- Feature dim: 2048 → 512
- ResNet frozen ban đầu, có thể fine-tune sau

### DecoderRNN
- Question encoder: LSTM (1 layer)
- Word embedding: 512 dim
- Hidden size: 512
- Fusion: Concatenate image + question features
- Classifier: 2-layer MLP → 1000 classes

### Training
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Scheduler: StepLR (step_size=5, gamma=0.1)
- Gradient clipping: max_norm=5.0

## Kết quả mong đợi

Với 5000 ảnh subset (~25,000 questions):
- Training time: ~1-2 giờ trên GPU T4
- Accuracy sau 10 epochs: ~40-50% (trên training set)
- Model size: ~100MB

## Troubleshooting

**Problem:** Runtime disconnected
- **Solution:** Lưu checkpoints thường xuyên vào Drive

**Problem:** Không tìm thấy image files
- **Solution:** Kiểm tra đường dẫn và format tên file (COCO_train2014_*.jpg)

**Problem:** Vocabulary size quá lớn
- **Solution:** Tăng `freq_threshold` trong Vocabulary class

**Problem:** Training quá chậm
- **Solution:** Đảm bảo đang dùng GPU runtime, giảm data size

## Tài liệu tham khảo

- VQA Dataset: https://visualqa.org/
- PyTorch Docs: https://pytorch.org/docs/
- Colab Guide: https://colab.research.google.com/notebooks/intro.ipynb

## License

Sử dụng cho mục đích học tập và nghiên cứu.
