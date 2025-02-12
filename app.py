import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch import nn
from custom_cnn import CNN  # Import your custom CNN model


# Define class labels
CLASS_LABELS = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Paths to model weights
CNN_MODEL_PATH = r"D:\Career\Cellula\Task 1\teeth_classification_model.pth"
RESNET_MODEL_PATH = r"D:\Career\Cellula\Task 1\best_model.pth"

def load_cnn_model(num_classes):
    try:
        model = CNN(num_classes)
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None
def load_resnet_model(num_classes):
    try:
        # Initialize ResNet50 model
        model = models.resnet50(pretrained=False)
        
        # Modify the first conv layer to match the checkpoint
        model.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        model.layer1[1].conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        # Modify the final fully connected layer
        model.fc = nn.Linear(2048, num_classes)
        
        # Load the state dict
        state_dict = torch.load(RESNET_MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ResNet model: {e}")
        return None

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def main():
    # Streamlit app title and description
    st.title("Teeth Disease Classification")
    st.write("Upload an image of teeth to classify the disease")

    # Dropdown menu to select the model
    model_choice = st.selectbox("Choose the model:", ("Custom CNN", "Pretrained ResNet"))

    # Load the appropriate model based on the selection
    model = None
    if model_choice == "Custom CNN":
        model = load_cnn_model(len(CLASS_LABELS))
    else:
        model = load_resnet_model(len(CLASS_LABELS))

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload a teeth image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load and display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            image_tensor = transform(image).unsqueeze(0)

            # Make prediction if model is loaded
            if model is not None:
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                # Display prediction and confidence
                predicted_class = CLASS_LABELS[predicted.item()]
                confidence = probabilities[0][predicted.item()].item() * 100
                
                # Create a result container
                result_container = st.container()
                with result_container:
                    st.success(f"Predicted Class: {predicted_class}")
                    st.info(f"Confidence: {confidence:.2f}%")
                    
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            st.write("Please try uploading a different image or check if the image format is supported.")

if __name__ == "__main__":
    main()