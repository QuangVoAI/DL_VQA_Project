# Code Citations

## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cggos/mlb/blob/5c8c00a672ff120cab1d65aa1a9287345cf53376/deep_learning/torch/dogs_vs_cats/3_transfer_learning_vgg.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/cooliotonyio/image-feature-search-engine/blob/58d9b3c98185db12c6224d01a96f3cdd927cfb70/search.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```


## License: unknown
https://github.com/pongmorrakot/copy_detect/blob/7d868029e25bc9aec7b0b4ecc63bf2c858fe6328/feature_extract.py

```
Đã đọc toàn bộ source code. Dưới đây là phân tích chuyên sâu theo từng hạng mục:

---

## 1. Kiến trúc nghiên cứu (8.5/10 → target 9.5+)

### Vấn đề hiện tại
- **`image_proj` không bao giờ được sử dụng**: Bạn khai báo `self.image_proj = nn.Linear(512, hidden_size)` trong `VQAModel` nhưng không gọi trong `forward()` — dead weight, tốn tham số.
- **No attention masking**: `BahdanauAttention` không mask padding tokens của question → attention leak vào `<PAD>`, đặc biệt nghiêm trọng khi câu hỏi ngắn mà batch có câu dài.
- **Teacher forcing schedule hardcoded**: `tf = max(0.5, 1.0 - (epoch-1)/epochs*0.5)` — tuyến tính, không có scheduled sampling curriculum chuẩn.
- **Beam search O(B×W×T)**: Vòng for qua từng sample trong batch → rất chậm. Không có length normalization penalty hay repetition penalty.

### Đề xuất cụ thể

```python
# (a) Fix BahdanauAttention với padding mask
class BahdanauAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask <PAD>
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

# (b) Tạo padding mask trong QuestionEncoder
def forward(self, questions, lengths):
    # ...existing...
    mask = (questions != 0).float()  # (B, T) — 1 for real, 0 for PAD
    return outputs, (h, c), mask     # pass mask to decoder

# (c) Beam search: thêm length normalization
# Thay vì: sc + v.item()
# Dùng: (sc + v.item()) / ((5 + len(seq)+1)**0.6 / (5+1)**0.6)  # Wu et al. 2016
```

- **Thêm METEOR metric** (standard VQA/NLG metric, thiếu trong paper submissions):
```python
from nltk.translate.meteor_score import meteor_score
# METEOR quan trọng hơn BLEU cho VQA vì xem xét synonyms + stemming
```

- **Xóa `image_proj`** hoặc thực sự sử dụng nó để project image features trước khi concat vào decoder.

---

## 2. Code Organization (6.5/10 → target 9+)

### Vấn đề hiện tại
- **Monolithic notebook**: 1242 dòng trong 1 file `.ipynb`, trộn lẫn data loading, model definition, training, evaluation, visualization.
- **Không thể import lại**: Ai muốn dùng `VQAModel` phải copy cell, không có Python module.
- **Không có config management**: Hyperparameters rải rác khắp notebook (`BATCH_SIZE=64` ở cell khác, `EPOCHS=20` ở cell khác, `DROPOUT=0.3` ở cell khác).

### Đề xuất cấu trúc project

```
DL_VQA_Project/
├── configs/
│   └── default.yaml          # TẤT CẢ hyperparams 1 chỗ
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py         # AOKVQA_Dataset, Vocabulary, collate_fn
│   │   ├── preprocessing.py   # normalize_answer, expand_data_with_rationales
│   │   └── glove.py           # download_glove, load_glove_embeddings
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder, QuestionEncoder
│   │   ├── decoder.py         # AnswerDecoder, BahdanauAttention
│   │   ├── vqa_model.py       # VQAModel (compose encoder + decoder)
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # train_model(), train loop
│   │   └── scheduler.py       # custom schedulers nếu cần
│   ├── evaluation/
│   │   ├── metrics.py         # compute_f1, compute_bleu, batch_metrics
│   │   └── inference.py       # beam_search standalone, greedy
│   └── visualization/
│       ├── plots.py           # training curves, bar chart
│       └── attention.py       # visualize_attention
├── notebooks/
│   └── VQA.ipynb              # CHỈ orchestration: import + run + display
├── scripts/
│   ├── train.py               # CLI training: python train.py --config configs/default.yaml
│   └── evaluate.py            # CLI eval: python evaluate.py --checkpoint best_M4.pth
├── checkpoints/
├── data/
├── requirements.txt
├── setup.py                   # hoặc pyproject.toml
└── README.md
```

### Config management với YAML

```yaml
# configs/default.yaml
seed: 42
data:
  hf_id: "HuggingFaceM4/A-OKVQA"
  train_ratio: 0.85
  freq_threshold: 2
  expand_rationales: true

model:
  embed_size: 300
  hidden_size: 512
  num_layers: 2
  dropout: 0.3
  glove_dim: 300

training:
  epochs: 20
  lr: 3e-4
  batch_size: 64
  label_smoothing: 0.1
  grad_clip: 5.0
  tf_start: 1.0
  tf_end: 0.5

eval:
  beam_width: 5
  max_len: 40
```

```python
# Trong code:
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """Type-safe config with validation."""
    seed: int = 42
    # ... load from yaml
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

---

## 3. Clean Structure (6/10 → target 9+)

### Vấn đề hiện tại
- **Import grouping**: `import os, io, re, string, random, math` — gom nhiều thư viện 1 dòng, vi phạm PEP 8.
- **Magic numbers**: `[-10]` trong freeze layers, `0.6` trong GloVe init, `3.0` trong VQA accuracy — nên là named constants.
- **Side effects trong cell**: `nltk.download()`, `device = get_device()`, `print(...)` chạy khi import → khó test.
- **Không có type hints** đầy đủ cho các hàm chính.
- **Comments tiếng Anh lẫn tiếng Việt** trong cùng codebase.

### Đề xuất cụ thể

```python
# (a) Constants tập trung
# src/constants.py
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
GLOVE_INIT_STD = 0.6
RESNET_FREEZE_CUTOFF = -10  # freeze all but last 10 params
VQA_ACC_DENOMINATOR = 3.0

# (b) Type hints cho core functions
from typing import Dict, List, Optional, Tuple

def batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    ...

def compute_bleu(
    pred: str,
    ref: str,
    max_n: int = 4,
) -> Dict[str, float]:
    ...

# (c) Tách side effects
# KHÔNG: device = get_device()  (top-level)
# CÓ:
def setup_environment(seed: int = 42) -> torch.device:
    """Set seeds and return device. Call explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # reproducibility > speed
    return get_device()
```

---

## 4. Experiment Design (9/10 → target 9.5+)

### Vấn đề hiện tại
- **Thiếu `val_bleu2`, `val_bleu3` trong training history** (đã phát hiện trước đó) — `batch_metrics()` tính nhưng không lưu.
- **Không log learning rate schedule** — không thể debug scheduler behavior.
- **Không có early stopping** — chỉ save best F1, nhưng vẫn train đủ 20 epochs dù có thể đã overfit.
- **Thiếu statistical significance**: Chỉ 1 run/model, không có error bars hay confidence intervals.
- **BLEU smoothing method1 có bias** cho câu ngắn — nên dùng `method4` (exponential smoothing) cho VQA.

### Đề xuất cụ thể

```python
# (a) Thêm early stopping
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# (b) Multi-seed runs cho significance testing
SEEDS = [42, 123, 456]
all_results = {seed: {} for seed in SEEDS}
for seed in SEEDS:
    setup_environment(seed)
    for name, cfg in model_configs.items():
        # ... train & eval
        all_results[seed][name] = test_metrics

# Report: mean ± std across seeds
for name in model_configs:
    f1s = [all_results[s][name]["f1"] for s in SEEDS]
    print(f"{name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# (c) Fix training history — thêm bleu2, bleu3
history = {"train_loss": [], "val_loss": [],
           "val_acc": [], "val_em": [], "val_f1": [],
           "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
           "lr": []}  # ← track LR schedule

# Trong loop:
history["val_bleu2"].append(m["bleu2"])
history["val_bleu3"].append(m["bleu3"])
history["lr"].append(optimizer.param_groups[0]["lr"])
```

---

## 5. Visualization (7.5/10 → target 9+)

### Vấn đề hiện tại
- **8 subplots trong 1 figure** — quá đông, khó đọc.
- **Bar chart cuối** dùng quá nhiều bars (7 metrics × 4 models = 28 bars) — unreadable.
- **Attention map**: Chỉ 3 samples cố định, không chọn representative cases (easy/hard/failure).
- **Không có confusion analysis** — không biết model sai ở đâu, loại câu hỏi nào.

### Đề xuất cụ thể

```python
# (a) Tách visualization thành focused figures
# Figure 1: Loss curves only (train + val, 4 models)
# Figure 2: Core metrics (Acc, EM, F1) — 1 row, 3 cols
# Figure 3: BLEU progression — 1 row, 4 cols
# Figure 4: Radar chart cho test comparison (thay vì bar chart)

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(test_results: dict, metrics: list, labels: list):
    """Radar/spider plot — much clearer for multi-metric comparison."""
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, m in test_results.items():
        values = [m[k] for k in metrics] + [m[metrics[0]]]
        ax.plot(angles, values, 'o-', label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison (Test Set)", fontsize=14, pad=20)
    return fig

# (b) Error analysis — breakdown by question type
def error_analysis(preds, refs, questions):
    """Categorize errors by question word (what/where/why/how/...)."""
    q_types = {}
    for p, r, q in zip(preds, refs, questions):
        first_word = q.split()[0].lower()
        if first_word not in q_types:
            q_types[first_word] = {"correct": 0, "total": 0}
        q_types[first_word]["total"] += 1
        if normalize_answer(p) == normalize_answer(r):
            q_types[first_word]["correct"] += 1
    
    for qw, stats in sorted(q_types.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        print(f"  {qw:>8s}: {acc:.1%}  ({stats['total']} questions)")

# (c) Qualitative: cherry-pick best/worst/interesting cases
def show_best_worst(preds, refs, questions, n=3):
    """Show top-n best predictions + top-n worst failures."""
    scored = [(compute_f1(p, r), i) for i, (p, r) in enumerate(zip(preds, refs))]
    scored.sort(reverse=True)
    print("=== BEST ===")
    for _, i in scored[:n]: ...
    print("=== WORST ===")
    for _, i in scored[-n:]: ...
```

---

## 6. Production Readiness (5/10 → target 8+)

### Vấn đề hiện tại
- **Không có logging** — chỉ `print()`, mất log khi notebook restart.
- **Không export model** — chỉ save `state_dict`, không có inference pipeline standalone.
- **Không có input validation** — model nhận raw tensor, không check shapes.
- **Không có error handling** — `Image.new("RGB", (224, 224))` khi fail → silent wrong data.
- **Không reproducible 100%**: Thiếu `torch.backends.cudnn.deterministic = True`.
- **Memory leak tiềm ẩn**: Beam search tạo list of tensors mỗi step, không release.
- **Không có versioning** cho data/model artifacts.

### Đề xuất cụ thể

```python
# (a) Structured logging thay thế print()
import logging
from datetime import datetime

def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler — persistent
    fh = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger

# (b) Inference pipeline — standalone, no notebook dependency
class VQAInferencePipeline:
    """Production-grade inference. Load once, predict many."""
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu"):
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]
        self.device = torch.device(device)
        
        self.model = VQAModel(...)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    @
```

