import streamlit as st
import torch.nn as nn

import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
num_classes = 7
model = CNN(num_classes)
# Load the trained model weights
model_path = r"D:\Career\Cellula\Task 1\teeth_classification_model.pth"  # Ensure the path is correct
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit app
st.title("Teeth Disease Classification")
st.write("Upload an image to classify.")

# Upload image using Streamlit
uploaded_file = st.file_uploader("Choose a teeth image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  # Get the class with the highest score

    # Define class labels
    class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
    predicted_class = class_labels[predicted.item()]

    # Display the result
    st.write(f"**Predicted Class:** {predicted_class}")
