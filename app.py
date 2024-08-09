import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from PIL import Image, ImageEnhance
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set up SQLAlchemy
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

# Load your trained model
try:
    model = load_model('model/malaria_model.h5')
    malaria_model = ResNet50(weights='imagenet')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error("Error loading model. Please check the model path and file.")

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
    # Load the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (heatmap.shape[0], heatmap.shape[1]))

    # Expand heatmap dimensions to (height, width, 1)
    heatmap = tf.expand_dims(heatmap, axis=-1)

    # Normalize the heatmap between 0 and 1
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize_with_crop_or_pad(heatmap, img.shape[0], img.shape[1])

    # Convert heatmap to RGB
    heatmap = tf.image.grayscale_to_rgb(heatmap)

    # Apply colormap using matplotlib and convert to Tensor
    heatmap = tf.convert_to_tensor(plt.cm.jet(heatmap.numpy())[:, :, :3], dtype=tf.float32)

    # Normalize the original image between 0 and 1
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    # Ensure heatmap and image have compatible shapes and types
    heatmap = tf.image.resize(heatmap, [img.shape[0], img.shape[1]])

    # Superimpose the heatmap on the original image
    superimposed_img = heatmap * alpha + img

    # Convert back to uint8
    superimposed_img = tf.image.convert_image_dtype(superimposed_img, dtype=tf.uint8)

    # Save and display the image
    superimposed_img = Image.fromarray(superimposed_img.numpy())
    superimposed_img.save(cam_path)
    st.image(superimposed_img, caption='Grad-CAM Image', use_column_width=True)

# Function to make predictions
def predict_malaria(img_path):
    try:
        logging.info(f"Loading image from path: {img_path}")
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        logging.info("Image loaded and preprocessed successfully.")

        prediction = model.predict(img_array)
        logging.info(f"Prediction made successfully: {prediction}")
        return prediction[0][0], img_array
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error(f"Error during prediction: {e}")
        return None, None

# Function to display confidence score
def display_confidence_score(prediction):
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Positive" if prediction > 0.5 else "Negative"
    
    fig, ax = plt.subplots()
    ax.barh([0], [confidence], color='red' if prediction > 0.5 else 'green')
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Confidence %')
    ax.set_title(f'Prediction: {label}')
    
    st.pyplot(fig)

# Function to generate report
def generate_report(prediction, img_path, heatmap):
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Positive" if prediction > 0.5 else "Negative"
    st.markdown("## Detailed Report")
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown("### Grad Cam Image")
    display_gradcam(img_path, heatmap)
    st.markdown("**General Information about Malaria Detection:**")
    st.markdown("""
    - Malaria is a life-threatening disease caused by parasites.
    - It is transmitted to people through the bites of infected female Anopheles mosquitoes.
    - Early diagnosis and treatment of malaria reduces disease and prevents deaths.
    """)

# Initialize session state for app_mode
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Home'

# Sidebar navigation with arrow buttons
st.sidebar.title('Navigation')
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

    home_image_path = 'assets/home_page.jpeg'
    if os.path.exists(home_image_path):
        st.image(home_image_path, use_column_width=True)
    else:
        st.warning("Home image not found. Please ensure 'home_image.jpg' is in the working directory.")

elif app_mode == 'Detect Malaria':
    st.title('Malaria Disease Detection')
    st.write('Upload an image of a blood smear to check for malaria.')

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(file_details)

        if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
            st.error("File size should be less than 5MB.")
        else:
            st.write("Classifying...")

            img_folder = 'images/uploaded_images/'
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            
            img_path = os.path.join(img_folder, uploaded_file.name)
            with open(img_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            img_Array = get_img_array(img_path, size=(224, 224))
            last_conv_layer_name = "conv5_block3_out"

            prediction, img_array = predict_malaria(img_path)

            if prediction is not None:
                st.image(image.array_to_img(img_array[0]), caption='Uploaded Image', use_column_width=True)
                display_confidence_score(prediction)
                heatmap = make_gradcam_heatmap(img_Array, malaria_model, last_conv_layer_name)
                generate_report(prediction, img_path, heatmap)

                feedback = st.radio("Is this prediction correct?", ("Yes", "No"))
                if st.button("Submit Feedback"):
                    feedback_bool = True if feedback == "Yes" else False
                    feedback_entry = Feedback(image_path=img_path, feedback=feedback_bool, prediction=float(prediction))
                    session.add(feedback_entry)
                    session.commit()
                    st.write("Thank you for your feedback!")
                    st.write("Feedback stored successfully!")

elif app_mode == 'About Malaria':
    st.title("About Malaria")
    st.markdown("""
    Malaria is a serious and sometimes fatal disease caused by a parasite that commonly infects a certain type of mosquito which feeds on humans. 
    People who get malaria are typically very sick with high fevers, shaking chills, and flu-like illness.
    **Symptoms of Malaria:**
    - Fever
    - Chills
    - Headache
    - Nausea and vomiting
    - Muscle pain and fatigue
    **Prevention and Treatment:**
    - Avoid mosquito bites by using mosquito repellent, sleeping under a bed net, and wearing protective clothing.
    - Antimalarial drugs are available to prevent and treat malaria.
    - Early diagnosis and prompt treatment are essential for reducing malaria-related deaths.
    """)
    malaria_image_path = 'assets/malaria_about.jpeg'
    if os.path.exists(malaria_image_path):
        st.image(malaria_image_path, use_column_width=True)
    else:
        st.warning("Malaria image not found. Please ensure 'malaria_about.jpeg' is in the working directory.")
