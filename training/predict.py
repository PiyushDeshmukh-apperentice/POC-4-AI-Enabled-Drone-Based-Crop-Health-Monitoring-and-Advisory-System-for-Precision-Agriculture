"""
Single image inference for rice paddy growth stage classification.

Usage:
  python predict.py --image path/to/your_image.jpg
  python predict.py --image path/to/your_image.jpg --model checkpoints/checkpoint_round1.pt
  python predict.py --image path/to/your_image.jpg --model checkpoints/checkpoint_enhanced_ann.pt
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

CLASSES_4 = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASSES_5 = ["Seedling", "Vegetative", "Flowering", "Mature", "Not_Rice"]
SYMBOLS_4 = ["[S]", "[V]", "[F]", "[M]"]
SYMBOLS_5 = ["[S]", "[V]", "[F]", "[M]", "[X]"]
CKPT_DIR = Path("checkpoints")
DEFAULT_CKPT = CKPT_DIR / "checkpoint_5class_finetune.pt"

# ── Model Definitions ────────────────────────────────────────────────────────
class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_b0()
        self.features = base.features
        self.avgpool  = base.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 384), nn.LayerNorm(384), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(384, 128),  nn.LayerNorm(128),  nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))

def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    input_size = ckpt.get("input_size", 224)

    if any(k.startswith("head.") for k in state.keys()):
        num_classes = state.get("head.9.weight", state.get("head.7.weight", torch.zeros(4))).shape[0]
        class_list = CLASSES_5 if num_classes == 5 else CLASSES_4
        symbol_list = SYMBOLS_5 if num_classes == 5 else SYMBOLS_4
        model = EnhancedRiceClassifier(num_classes=num_classes)
        arch = f"Enhanced Fine-Tuned ANN ({num_classes}-Class)"
    else:
        num_classes = 4
        class_list = CLASSES_4
        symbol_list = SYMBOLS_4
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        arch = "Baseline Transfer Learning"

    model.load_state_dict(state)
    model.to(device).eval()
    return model, input_size, arch, class_list, symbol_list

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_path: Path, ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, input_size, arch, class_list, symbol_list = load_model(ckpt_path, device)

    # Load & preprocess image
    img = Image.open(image_path)
    img.draft("RGB", (input_size, input_size))
    img = img.convert("RGB")

    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(img).unsqueeze(0).to(device)  # Add batch dimension

    # Run through model
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx  = probs.argmax().item()
    pred_class = class_list[pred_idx]
    confidence = probs[pred_idx].item() * 100

    # Print result
    print()
    print("=" * 50)
    print(f"  Image    : {image_path.name}")
    print(f"  Model    : {arch}")
    print(f"  Device   : {device}")
    print("=" * 50)
    print(f"\n  PREDICTION:  {symbol_list[pred_idx]}  {pred_class.upper()}  ({confidence:.1f}% confident)\n")
    print("  All Class Probabilities:")
    for i, (cls, sym) in enumerate(zip(class_list, symbol_list)):
        bar = "#" * int(probs[i].item() * 30)
        print(f"    {sym} {cls:<12} {probs[i].item()*100:5.1f}%  {bar}")
    print("=" * 50)

    return pred_class, confidence

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Predict rice paddy growth stage from a single image")
    ap.add_argument("--image",  type=Path, required=True, help="Path to JPG/PNG image")
    ap.add_argument("--model",  type=Path, default=DEFAULT_CKPT,
                    help=f"Path to checkpoint (default: {DEFAULT_CKPT})")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if not args.model.exists():
        # Fall back to round1 if enhanced not found
        fallback = CKPT_DIR / "checkpoint_round1.pt"
        if fallback.exists():
            print(f"Note: {args.model.name} not found. Using {fallback.name} instead.")
            args.model = fallback
        else:
            print(f"Error: No checkpoint found at {args.model}")
            sys.exit(1)

    predict(args.image, args.model)

if __name__ == "__main__":
    main()
