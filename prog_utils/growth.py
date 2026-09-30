from collections import Counter
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_weights" / "growth_stage.pt"
CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
IMAGE_SIZE = (1280, 960)
REGION_SIZE = 320
MODEL_RES = 288
STAGE_COLORS = {
    "Seedling": (0, 181, 217),
    "Vegetative": (45, 106, 79),
    "Flowering": (247, 127, 0),
    "Mature": (255, 209, 102),
}


def load_growth_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = models.efficientnet_b1(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), device


def analyze_growth_image(image, model, device):
    import torch
    from PIL import Image, ImageDraw
    from torchvision import transforms

    image = image.convert("RGB")
    if image.size != IMAGE_SIZE:
        raise ValueError(
            f"Growth-stage analysis requires a {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} image; "
            f"received {image.width} x {image.height}."
        )

    transform = transforms.Compose([
        transforms.Resize((MODEL_RES, MODEL_RES)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    coordinates = [
        (x, y)
        for y in range(0, IMAGE_SIZE[1], REGION_SIZE)
        for x in range(0, IMAGE_SIZE[0], REGION_SIZE)
    ]
    tiles = [
        image.crop((x, y, x + REGION_SIZE, y + REGION_SIZE))
        for x, y in coordinates
    ]
    batch = torch.stack([transform(tile) for tile in tiles]).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()

    predictions = probabilities.argmax(axis=1)
    records = []
    overlay = image.convert("RGBA")
    color_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(color_layer)

    for region_id, ((x, y), prediction, scores) in enumerate(
        zip(coordinates, predictions, probabilities), start=1
    ):
        stage = CLASSES[int(prediction)]
        confidence = float(scores[prediction])
        color = STAGE_COLORS[stage]
        draw.rectangle(
            (x, y, x + REGION_SIZE - 1, y + REGION_SIZE - 1),
            fill=(*color, 65),
            outline=(*color, 255),
            width=3,
        )
        draw.rectangle((x + 8, y + 8, x + 178, y + 38), fill=(*color, 235))
        draw.text((x + 14, y + 14), f"R{region_id}: {stage} {confidence:.0%}", fill=(255, 255, 255, 255))
        records.append({
            "Region": region_id,
            "X_Min": x,
            "Y_Min": y,
            "X_Max": x + REGION_SIZE,
            "Y_Max": y + REGION_SIZE,
            "Growth_Stage": stage,
            "Confidence": round(confidence, 4),
        })

    stage_counts = Counter(record["Growth_Stage"] for record in records)
    summary = {
        "region_count": len(records),
        "majority_stage": stage_counts.most_common(1)[0][0],
        "stage_counts": dict(stage_counts),
    }
    return Image.alpha_composite(overlay, color_layer).convert("RGB"), records, summary
