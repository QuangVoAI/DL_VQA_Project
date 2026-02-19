"""CLI script: Run inference on images with trained VQA models.

Usage:
    # Single prediction
    python scripts/inference.py --image photo.jpg --question "What color is the car?"

    # Batch prediction from JSON
    python scripts/inference.py --batch inputs.json --output results.json

    # ONNX export
    python scripts/inference.py --export-onnx --model M4_Pretrained_Attn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image

from src.config import Config
from src.data.dataset import SOS_IDX, EOS_IDX
from src.models.vqa_model import VQAModel
from src.utils.helpers import get_device, decode_sequence


class VQAInferencePipeline:
    """Production-ready VQA inference pipeline.

    Features:
        - Load from checkpoint + vocabulary
        - Single or batch prediction
        - Beam search or greedy decoding
        - ONNX export support

    Args:
        checkpoint_path: Path to model checkpoint (.pth).
        vocab_path: Path to vocabulary file (.pth).
        device_str: Device string ("cpu", "cuda", "mps", "auto").
        model_kwargs: Extra kwargs for VQAModel constructor.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device_str: str = "auto",
        **model_kwargs,
    ) -> None:
        if device_str == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device_str)

        # Load vocabularies
        vocabs = torch.load(vocab_path, weights_only=False)
        self.q_vocab = vocabs["question_vocab"]
        self.a_vocab = vocabs["answer_vocab"]

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Determine model config
        q_size, a_size = len(self.q_vocab), len(self.a_vocab)
        default_kwargs = dict(
            embed_size=300, hidden_size=512, num_layers=2,
            dropout=0.0, use_pretrained_cnn=True, use_attention=True,
        )
        default_kwargs.update(model_kwargs)

        self.model = VQAModel(q_size, a_size, **default_kwargs)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(f"VQA Pipeline loaded on {self.device}")
        if "metrics" in ckpt:
            m = ckpt["metrics"]
            print(f"  Checkpoint metrics: F1={m.get('f1', '?'):.4f}  "
                  f"Acc={m.get('accuracy', '?'):.4f}")

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        beam_width: int = 5,
    ) -> str:
        """Predict answer for a single image + question pair.

        Args:
            image: PIL Image.
            question: Question string.
            beam_width: Beam search width (0 = greedy).

        Returns:
            Predicted answer string.
        """
        img = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        q_ids = [SOS_IDX] + self.q_vocab.numericalize(question) + [EOS_IDX]
        q_tensor = torch.tensor([q_ids]).to(self.device)
        q_len = torch.tensor([len(q_ids)])
        gen = self.model.generate(
            img, q_tensor, q_len,
            use_beam=(beam_width > 0), beam_width=max(beam_width, 1),
        )
        return decode_sequence(gen[0].cpu().tolist(), self.a_vocab)

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        beam_width: int = 5,
        batch_size: int = 32,
    ) -> list[str]:
        """Predict answers for a batch of (image, question) pairs.

        Args:
            images: List of PIL Images.
            questions: List of question strings.
            beam_width: Beam search width.
            batch_size: Processing batch size.

        Returns:
            List of predicted answer strings.
        """
        from torch.nn.utils.rnn import pad_sequence

        results = []
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_imgs = []
            batch_qs = []
            batch_qlens = []

            for i in range(start, end):
                img = self.transform(images[i].convert("RGB"))
                batch_imgs.append(img)
                q_ids = [SOS_IDX] + self.q_vocab.numericalize(questions[i]) + [EOS_IDX]
                batch_qs.append(torch.tensor(q_ids))
                batch_qlens.append(len(q_ids))

            imgs_t = torch.stack(batch_imgs).to(self.device)
            qs_t = pad_sequence(batch_qs, batch_first=True, padding_value=0).to(self.device)
            ql_t = torch.tensor(batch_qlens)

            gen = self.model.generate(
                imgs_t, qs_t, ql_t,
                use_beam=(beam_width > 0), beam_width=max(beam_width, 1),
            )

            for i in range(gen.size(0)):
                results.append(decode_sequence(gen[i].cpu().tolist(), self.a_vocab))

        return results

    def export_onnx(self, output_path: str = "vqa_model.onnx") -> None:
        """Export model encoder components to ONNX format.

        Note: Full seq2seq export is limited; exports the image encoder and
        question encoder for feature extraction. Full generation requires
        the custom beam search logic.

        Args:
            output_path: Output ONNX file path.
        """
        print("Exporting image encoder to ONNX ...")
        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)

        # Export image encoder
        img_encoder = self.model.image_encoder
        img_encoder.eval()
        torch.onnx.export(
            img_encoder,
            dummy_img,
            output_path.replace(".onnx", "_image_encoder.onnx"),
            input_names=["image"],
            output_names=["image_features"],
            dynamic_axes={"image": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"  Image encoder saved → {output_path.replace('.onnx', '_image_encoder.onnx')}")

        print("Note: Full seq2seq model with beam search is not fully ONNX-exportable.")
        print("      Use the PyTorch model for inference with beam search.")
        print("      ONNX export covers the image encoder for feature extraction.")


def main() -> None:
    parser = argparse.ArgumentParser(description="VQA Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_M4_Pretrained_Attn.pth")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab_aokvqa.pth")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--beam-width", type=int, default=5)

    # Single prediction
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--question", type=str, default=None, help="Question text")

    # Batch prediction
    parser.add_argument("--batch", type=str, default=None,
                        help="JSON file with list of {image, question} pairs")
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Output JSON file for batch predictions")

    # ONNX export
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export model to ONNX format")
    parser.add_argument("--onnx-path", type=str, default="exports/vqa_model.onnx")

    args = parser.parse_args()

    pipe = VQAInferencePipeline(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device_str=args.device,
    )

    if args.export_onnx:
        os.makedirs(os.path.dirname(args.onnx_path) or ".", exist_ok=True)
        pipe.export_onnx(args.onnx_path)
        return

    if args.image and args.question:
        img = Image.open(args.image)
        t0 = time.time()
        answer = pipe.predict(img, args.question, beam_width=args.beam_width)
        elapsed = time.time() - t0
        print(f"Q: {args.question}")
        print(f"A: {answer}")
        print(f"Time: {elapsed:.3f}s")

    elif args.batch:
        with open(args.batch) as f:
            items = json.load(f)

        images = [Image.open(item["image"]) for item in items]
        questions = [item["question"] for item in items]

        t0 = time.time()
        answers = pipe.predict_batch(images, questions, beam_width=args.beam_width)
        elapsed = time.time() - t0

        results = []
        for item, answer in zip(items, answers):
            results.append({**item, "prediction": answer})

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Processed {len(items)} samples in {elapsed:.2f}s")
        print(f"Results saved → {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
