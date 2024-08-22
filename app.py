import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import altair as alt
import logging

# Set Streamlit app configuration
st.set_page_config(
    page_title="Malaria Disease Detection",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the background image
background_image = """
<style>
body {
    background-image: url("Malaria_Disease.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}
</style>
"""

# Embed the CSS in the Streamlit app
st.markdown(background_image, unsafe_allow_html=True)

# Enable Altair dark theme
alt.themes.enable("dark")

# Set up SQLAlchemy for feedback
Base = declarative_base()

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String)
    feedback = Column(Boolean)
    prediction = Column(Float)

engine = create_engine('sqlite:///feedback.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Adding Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
    }
    .stRadio>div {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
try:
    malaria_model = load_model('model/malaria_model.h5')  # Custom malaria detection model
    resnet_model = ResNet50(weights='imagenet')  # ResNet50 model for Grad-CAM
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    st.error("Error loading models. Please check the model paths and files.")

def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

    return cam_path


def predict_malaria(img_path):
    try:
        logging.info(f"Loading image from path: {img_path}")
        img = image.load_img(img_path, target_size=(128, 128))  # Resize to 128x128 for malaria detection
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        logging.info("Image loaded and preprocessed successfully.")

        prediction = malaria_model.predict(img_array)
        logging.info(f"Prediction made successfully: {prediction}")
        return prediction[0][0], img_array
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error(f"Error during prediction: {e}")
        return None, None
    
def display_confidence_score(prediction, fig_height=4):
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Positive" if prediction > 0.5 else "Negative"
    
    fig, ax = plt.subplots(figsize=(5, fig_height))
    ax.barh([0], [confidence], color='red' if prediction > 0.5 else 'green')
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Confidence %')
    ax.set_title(f'Prediction: {label}')
    
    return fig

def display_results_and_gradcam(prediction, img_path, heatmap):
    confidence_fig = display_confidence_score(prediction, fig_height=4.5)
    cam_path = display_gradcam(img_path, heatmap)
    
    # Create columns for grid layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Uploaded Image")
        st.image(img_path, width=350)  # Using st.image to directly show the image
        
        st.pyplot(confidence_fig)
    
    with col2:
        st.markdown("### Grad-CAM Image")
        st.image(cam_path, width=350)  # Using st.image to directly show the image

def generate_report(prediction, img_path, heatmap):
    st.markdown("# Detailed Report")
    
    # Determine the result based on prediction
    result_label = "Positive" if prediction > 0.5 else "Negative"
    confidence_percentage = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    
    # Display the result and confidence with larger font sizes using HTML
    st.markdown(f"""
    <p style="font-size: 24px;"><strong>Result:</strong> The analysis indicates that the blood smear is <strong>{result_label}</strong>.</p>
    <p style="font-size: 24px;"><strong>Confidence Level:</strong> {confidence_percentage:.2f}%</p>
    """, unsafe_allow_html=True)
    
    # Display the uploaded image and Grad-CAM image
    display_results_and_gradcam(prediction, img_path, heatmap)
    
    # Provide general information about malaria detection
    st.markdown("## General Information About Malaria Detection")
    st.markdown("""
    - **Malaria** is a serious and sometimes fatal disease caused by parasites that enter the human body through the bites of infected mosquitoes.
    - **Transmission:** It is transmitted by the Anopheles mosquito, which is infected with the malaria parasites.
    - **Symptoms:** Symptoms include fever, chills, and flu-like illness. If not treated promptly, malaria can lead to severe illness and death.
    - **Prevention:** Using mosquito nets, insect repellents, and taking antimalarial medication can help prevent malaria.
    - **Diagnosis and Treatment:** Malaria is diagnosed through blood tests, and treatment involves antimalarial medications prescribed by a healthcare provider.
    - **Importance of Early Detection:** Early diagnosis and treatment are crucial to reduce the severity of the disease and prevent deaths.
    """)


# Initialize session state for app_mode
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Home'

# Sidebar navigation with arrow buttons
st.sidebar.title('Malaria Disease Detection App')
if st.sidebar.button('→ Home'):
    st.session_state.app_mode = 'Home'
if st.sidebar.button('→ Detect Malaria'):
    st.session_state.app_mode = 'Detect Malaria'
if st.sidebar.button('→ About Malaria'):
    st.session_state.app_mode = 'About Malaria'

# Set app_mode based on session state
app_mode = st.session_state.app_mode

if app_mode == 'Home':
    st.title('Welcome to the Malaria Disease Detection App')
    st.markdown("""
    This application uses a deep learning model to detect malaria from blood smear images. 
    You can upload an image and get a prediction of whether the blood smear is infected with malaria or not.
    Use the navigation bar to access different sections of the app.
    """)

    home_image_path = 'assets/home_page.jpg'
    if os.path.exists(home_image_path):
        image = Image.open(home_image_path)
        resized_image = image.resize((700, 300))  # Adjust these values as needed
        st.image(resized_image, use_column_width=True)
    else:
        st.warning("Home image not found. Please ensure 'home_page.png' is in the working directory.")

elif app_mode == 'Detect Malaria':
    st.title('Malaria Disease Detection')
    
    # First Row: Upload an image
    st.write('Upload an image of a blood smear to check for malaria.')
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.markdown("#### Uploaded Image Details")
        st.write(file_details)
        
        # Save uploaded file
        img_path = os.path.join("uploads", uploaded_file.name)
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img_array_resnet = get_img_array(img_path, size=(224, 224))
        last_conv_layer_name = "conv5_block3_out"  # Adjust this to the correct layer name for ResNet50

        with st.spinner('Classifying the image, please wait...'):
            prediction, img_array = predict_malaria(img_path)

            if prediction is not None:
                # Generate Grad-CAM heatmap
                heatmap = make_gradcam_heatmap(img_array_resnet, resnet_model, last_conv_layer_name)
                # Display results and Grad-CAM image in a grid layout
                generate_report(prediction, img_path, heatmap)

                # Collect and handle feedback
                feedback = st.radio("Is this prediction correct?", ("Yes", "No"))
                if st.button("Submit Feedback"):
                    feedback_bool = True if feedback == "Yes" else False
                    feedback_entry = Feedback(image_path=img_path, feedback=feedback_bool, prediction=float(prediction))
                    session.add(feedback_entry)
                    session.commit()
                    st.success("Thank you for your feedback! 🎉")
                    st.balloons()

elif app_mode == 'About Malaria':
    st.title('About Malaria')
    st.markdown("""
    Malaria is a serious and sometimes fatal disease caused by parasites that enter the human body through the bites of infected mosquitoes. 
    Symptoms include fever, chills, and flu-like illness. If not treated promptly, malaria can lead to severe illness and death.

    **Prevention Measures:**
    - Use mosquito nets and insect repellent.
    - Take antimalarial medication if recommended.
    - Ensure proper sanitation and drainage to prevent mosquito breeding.

    **Diagnosis and Treatment:**
    - Diagnosis is done through blood tests.
    - Treatment involves antimalarial medications prescribed by a healthcare provider.
    """)
    About_image_path = 'assets/prevent.jpeg'
    if os.path.exists(About_image_path):
        image = Image.open(About_image_path)
        resized_image = image.resize((700, 300))  # Adjust these values as needed
        st.image(resized_image, use_column_width=True)
    else:
        st.warning("About image not found. Please ensure 'home_page.png' is in the working directory.")

# Streamlit footer
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
