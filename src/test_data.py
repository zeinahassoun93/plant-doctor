from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import random

ROOT = Path("data/raw")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Train and valid (have class subfolders)
train_ds = datasets.ImageFolder(ROOT / "train", transform=transform)
val_ds   = datasets.ImageFolder(ROOT / "valid", transform=transform)

print("Classes:", train_ds.classes)
print("Num classes:", len(train_ds.classes))
print("Train size:", len(train_ds))
print("Val size:", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

images, labels = next(iter(train_loader))
print("Batch images shape:", images.shape)   # expect [8, 3, 224, 224]
print("Batch labels:", labels)

# Visualize first 4 samples
fig, axes = plt.subplots(1, 4, figsize=(12,3))
for i in range(4):
    img = images[i].permute(1,2,0)  # C,H,W -> H,W,C
    axes[i].imshow(img)
    axes[i].set_title(train_ds.classes[labels[i]])
    axes[i].axis("off")
plt.show()

# --- Handle test folder (flat images, no labels) ---
test_folder = ROOT / "test"
if test_folder.exists():
    test_imgs = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.png"))
    print(f"Test images found: {len(test_imgs)}")
    if test_imgs:
        # show a few random test images
        fig, axes = plt.subplots(1, 4, figsize=(12,3))
        for i, ax in enumerate(axes):
            img_path = random.choice(test_imgs)
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_title("test img")
            ax.axis("off")
        plt.show()
