import torch
import argparse
from torchvision import models, transforms, datasets
from PIL import Image

def predict_animal(img_path, model_path, num_classes, data_dir):
    # transforming images to the same format as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization by ImageNet parameters
    ])

    model = models.resnet18(pretrained=False)  # loading without pre-trained weights
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  

    img = Image.open(img_path)

    # apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)  

    with torch.no_grad():  # disable gradient calculation
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)  # get the class index with maximum probability

    # get class from index
    class_idx = predicted.item()

    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = train_dataset.classes
    return {class_names[class_idx]}

def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on an image classification model")
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to classify')
    parser.add_argument('--model_path', type=str, default='models/animal_recognition_model.pth', help='Path to the trained model file')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--data_dir', type=str, default='data/animals', help='Directory of the dataset (with ImageFolder structure)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    detected_animal = predict_animal(args.img_path, args.model_path, args.num_classes, args.data_dir)

    print(f"Predicted class name: {detected_animal}")