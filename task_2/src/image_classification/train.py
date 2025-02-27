import torch
import torchvision.transforms as transforms
import torch.profiler
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import argparse

def train_model(data_dir, model_save_path, num_epochs=5, batch_size=96, learning_rate=0.001):
    # image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize images to fit the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # data loading
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = models.resnet18(pretrained=True)

    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # move data to GPU (if available)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # save model
    torch.save(model.state_dict(), model_save_path)
    print(f"model saved to {model_save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification model")
    parser.add_argument('--data_dir', type=str, default='data/animals', help='Path to the dataset')
    parser.add_argument('--model_save_path', type=str, default='models/animal_recognition_model.pth', help='Path to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args.data_dir, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate)
