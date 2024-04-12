import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_image(image, target_size=(100, 100)):
    """
    Preprocess the image to fit the model's input requirements.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    # Normalize the image
    image_array = np.array(image) / 255.0
    # Add a batch dimension
    return np.expand_dims(image_array, axis=0)


# Load your trained model
model = load_model('outputs/v1/powdery_mildew_detector_model.keras')


def page_powdery_mildew_detector_body():
    """
    Function to display the powdery mildew detector page content.
    Allows users to upload images and see the model's predictions.
    """
    st.write("### Powdery Mildew Detection in Cherry Leaves")

    st.info("""
    For additional information, please visit and **read** the
    [Project README file](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/blob/main/README.md)
    """)

    uploaded_file = st.file_uploader("Choose a cherry leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf', use_column_width=True)

        if st.button('Predict'):
            preprocessed_image = preprocess_image(image)
            pred_proba = model.predict(preprocessed_image)[0]
            pred_class = 'powdery mildew' if pred_proba > 0.5 else 'healthy'
            st.success(f"The leaf is predicted to be {pred_class} with probability {pred_proba[0]:.2f}.")