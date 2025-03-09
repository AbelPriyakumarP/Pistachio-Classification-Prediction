import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os

class_names = ['Kirmizi_Pistachio','Siirt_Pistachio']
trained_model = None

class PistachioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(64*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transform(image).unsqueeze(0).to(device)

    global trained_model
    if trained_model is None:
        trained_model = PistachioCNN(num_classes=2).to(device)
        model_path = os.path.join("model", "saved_model.pth")
        trained_model.load_state_dict(torch.load(model_path))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]