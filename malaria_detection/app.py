import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load your trained model
model = load_model('/home/pankaj/Desktop/Projects/task1/Malaria_disease_detection/malaria_detection/model/malaria_model.h5')

# Function to make predictions
def predict_malaria(img_path):
    # Adjust target size to match model input shape
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction[0][0]

# Streamlit app
st.title('Malaria Disease Detection')

st.write('Upload an image of a blood smear to check for malaria.')

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_folder = 'images/uploaded_images/'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    img_path = os.path.join(img_folder, 'sample_image.png')
    with open(img_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    prediction = predict_malaria(img_path)

    if prediction > 0.5:
        st.write("The image is predicted to be Positive for Malaria with a confidence of {:.2f}%".format(prediction * 100))
    else:
        st.write("The image is predicted to be Negative for Malaria with a confidence of {:.2f}%".format((1 - prediction) * 100))
