import torch
from torchvision import transforms, datasets
from PIL import Image
import timm
from pathlib import Path
import pandas as pd


root_directory = Path("data/raw")
train_directory = root_directory / "train"
test_directory = root_directory / "test"
model_path = Path("models/best_model.pth")
output_csv_path = Path("models/predictions.csv")

checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
train_ds = datasets.ImageFolder(train_directory)
classes = train_ds.classes
print("Classes loaded:", len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=len(classes))
state_dict = torch.load(model_path, map_location=device, weights_only=True) 
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_imgs = list(test_directory.glob("*.jpg")) + list(test_directory.glob("*.png"))
print(f"Found {len(test_imgs)} test images.")

results = []
with torch.no_grad():
    for img_path in test_imgs:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()
        pred_class = classes[pred_idx]
        results.append({"filename": img_path.name, "prediction": pred_class})

# --- Save to CSV ---
df = pd.DataFrame(results)
output_csv_path.parent.mkdir(exist_ok=True)
df.to_csv(output_csv_path, index=False)
print(f"✅ Predictions saved to {output_csv_path}")
