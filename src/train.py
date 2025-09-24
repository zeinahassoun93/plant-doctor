import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm 
from pathlib import Path
from sklearn.metrics import classification_report

root_directory = Path("data/raw")
train_directory = root_directory / "train"
valid_directory = root_directory / "valid"
output_model_path = Path("models")

image_size = 224
batch_size = 32
num_epochs = 5
learning_rate = 0.001
num_classes = 38
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_directory, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_directory, transform=valid_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", train_dataset.classes)
num_classes = len(train_dataset.classes)
print("Num classes:", num_classes)   



model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


best_val_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    print("Epoch", epoch+1,"starting...") 
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        output_model_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_model_path / "best_model.pth")
        print("Best model saved.")



print("Training complete.")
print("Best Validation Accuracy:", best_val_accuracy)
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))