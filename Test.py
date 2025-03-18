import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn



class CNNClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super(CNNClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

def predict_disease(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image pre-processing to match the input requirements of the model
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust if needed
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Load the trained model
    model = CNNClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)

    # Convert predicted index to class label
    predicted_class = class_names[predicted_idx.item()]
    confidence = predicted_prob.item() * 100

    # Plot the image along with the predicted class and confidence
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Predicted Disease: {predicted_class}\nConfidence: {confidence:.2f}%')
    plt.show()

    # Display Top 3 predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    print("\nTop 3 Predictions:")
    for i in range(3):
        print(f"{class_names[top_indices[0][i]]}: {top_probs[0][i]*100:.2f}%")

    return predicted_class, confidence

# Example usage:
image_path = r'SampleData\0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.jpg'  # Adjust with your image path
model_path = 'final_model.pth'  # Path to your saved model
class_names = [
     'Pepper_bell__Bacterial_spot',
'Pepper_bell__healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite',
'Tomato__Target_Spot',
'Tomato_Tomato_YellowLeaf_Curl_Virus',
'Tomato__Tomato_mosaic_virus',
'Tomato_healthy'
] # Replace with actual class names

predicted_class, confidence = predict_disease(image_path, model_path, class_names)