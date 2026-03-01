import io
import os
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.models.vqa_model import VQAModel
from src.utils.helpers import decode_sequence


app = FastAPI(title="VQA API", version="1.0.0")

# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class VQAServerState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, VQAModel] = {}
        self.q_vocab = None
        self.a_vocab = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )


state = VQAServerState()


@app.on_event("startup")
def load_models() -> None:
    """
    Nạp artifact đã được tạo từ notebook: vqa_deploy_all_models.pth
    Artifact cần có:
      - config
      - model_states: dict[name -> state_dict]
      - q_vocab, a_vocab
    """
    artifact_path = "vqa_deploy_all_models.pth"
    if not os.path.exists(artifact_path):
        raise RuntimeError(
            f"Không tìm thấy {artifact_path}. Hãy chạy cell notebook để tạo artifact trước."
        )

    artifact = torch.load(artifact_path, map_location=state.device, weights_only=False)
    cfg_dict: dict = artifact.get("config", {})
    model_states: Dict[str, dict] = artifact.get("model_states", {})
    state.q_vocab = artifact.get("q_vocab")
    state.a_vocab = artifact.get("a_vocab")

    if not cfg_dict or not model_states or state.q_vocab is None or state.a_vocab is None:
        raise RuntimeError("Artifact không đầy đủ thông tin cần thiết để deploy.")

    # Lấy các cấu hình cần thiết trực tiếp từ dict
    model_cfg = cfg_dict.get("model", {})
    model_variants: dict = cfg_dict.get("model_variants", {})

    # Dựng lại từng model
    for name, variant_cfg in model_variants.items():
        if name not in model_states:
            continue

        model = VQAModel(
            q_vocab_size=len(state.q_vocab),
            a_vocab_size=len(state.a_vocab),
            embed_size=model_cfg.get("embed_size", 300),
            hidden_size=model_cfg.get("hidden_size", 512),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            q_pretrained_emb=None,
            a_pretrained_emb=None,
            **variant_cfg,
        ).to(state.device)

        model.load_state_dict(model_states[name], strict=False)
        model.eval()
        state.models[name] = model

    if not state.models:
        raise RuntimeError("Không có model nào được nạp từ artifact.")


@app.post("/v1/predict")
async def predict(
    question: str = Form(..., description="Câu hỏi VQA"),
    model_name: Optional[str] = Form(
        None,
        description=(
            "Tên model: M1_Scratch_NoAttn, M2_Scratch_Attn, "
            "M3_Pretrained_NoAttn, M4_Pretrained_Attn. Để trống sẽ chạy tất cả."
        ),
    ),
    image: UploadFile = File(..., description="Ảnh input (JPEG/PNG)"),
) -> JSONResponse:
    if not state.models:
        raise HTTPException(status_code=500, detail="Models chưa được nạp.")

    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được file ảnh.")

    img_tensor = state.transform(pil_img).unsqueeze(0).to(state.device)

    # Chuẩn bị câu hỏi
    q_tokens = (
        [state.q_vocab.stoi["<SOS>"]]
        + state.q_vocab.numericalize(question)
        + [state.q_vocab.stoi["<EOS>"]]
    )
    q_tensor = torch.tensor(q_tokens).unsqueeze(0).to(state.device)
    q_len = torch.tensor([len(q_tokens)])

    # Chọn model
    target_models: Dict[str, VQAModel]
    if model_name:
        if model_name not in state.models:
            raise HTTPException(status_code=400, detail="model_name không hợp lệ.")
        target_models = {model_name: state.models[model_name]}
    else:
        target_models = state.models

    results: Dict[str, str] = {}
    for name, model in target_models.items():
        with torch.no_grad():
            gen = model.generate(
                img_tensor,
                q_tensor,
                q_len,
                use_beam=True,
                beam_width=5,
            )
        answer = decode_sequence(gen[0].cpu().tolist(), state.a_vocab)
        results[name] = answer

    return JSONResponse(
        {
            "question": question,
            "answers": results,
            "models": list(results.keys()),
        }
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "num_models": str(len(state.models))}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend index.html not found.")
    return FileResponse(index_path)

