import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Load your trained model
model = load_model('model/malaria_model.h5')

# Function to make predictions
def predict_malaria(img_path):
    # Adjust target size to match model input shape
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction[0][0], img_array

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
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
    heatmap = heatmap.numpy()
    return heatmap

# Function to save and display the heatmap
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)
    return cam_path

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
    
    prediction, img_array = predict_malaria(img_path)
    
    if prediction > 0.5:
        st.write("The image is predicted to be Positive for Malaria with a confidence of {:.2f}%".format(prediction * 100))
    else:
        st.write("The image is predicted to be Negative for Malaria with a confidence of {:.2f}%".format((1 - prediction) * 100))

    last_conv_layer_name = "conv2d_2"  # Change this to the name of the last convolutional layer in your model
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    cam_path = save_and_display_gradcam(img_path, heatmap)

    st.image(cam_path, caption='Grad-CAM Heatmap.', use_column_width=True)
