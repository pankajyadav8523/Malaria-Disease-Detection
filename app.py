import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from PIL import Image, ImageEnhance
import sqlite3
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
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error("Error loading model. Please check the model path and file.")

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
def generate_report(prediction, img_path):
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Positive" if prediction > 0.5 else "Negative"
    
    st.markdown("## Detailed Report")
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown("**General Information about Malaria Detection:**")
    st.markdown("""
    - Malaria is a life-threatening disease caused by parasites.
    - It is transmitted to people through the bites of infected female Anopheles mosquitoes.
    - Early diagnosis and treatment of malaria reduces disease and prevents deaths.
    """)

# Streamlit app
st.sidebar.title('Navigation')
app_mode = st.sidebar.selectbox('Choose the app mode', ['Home', 'Detect Malaria', 'About Malaria'])

if app_mode == 'Home':
    st.title('Welcome to the Malaria Disease Detection App')
    st.markdown("""
    This application uses a deep learning model to detect malaria from blood smear images. 
    You can upload an image and get a prediction of whether the blood smear is infected with malaria or not.
    Use the navigation bar to access different sections of the app.
    """)
    st.image('home_image.jpg', use_column_width=True)

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
            
            prediction, img_array = predict_malaria(img_path)

            if prediction is not None:
                st.image(image.array_to_img(img_array[0]), caption='Uploaded Image', use_column_width=True)
                display_confidence_score(prediction)
                generate_report(prediction, img_path)
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
    **Prevention Methods:**
    - Use insect repellent
    - Sleep under a mosquito net
    - Take antimalarial drugs if traveling to a high-risk area
    - Wear long sleeves and pants to prevent mosquito bites
    """)
    st.image('malaria_prevention.jpg', use_column_width=True)

st.sidebar.title('Contact Us')
st.sidebar.info('For any inquiries or support, please contact us at: support@malariadetection.com')
