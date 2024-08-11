import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import joblib

st.title('Image Classification with CNN')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    

    class CNNModel(torch.nn.Module):
    
        def __init__(self):
            super(CNNModel, self).__init__()          

        def forward(self, x):
            return x

    model = joblib.load(open('model.pkl', 'rb'))
    
    # Perform prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        st.write(f'Predicted Category: {predicted.sum()}')
