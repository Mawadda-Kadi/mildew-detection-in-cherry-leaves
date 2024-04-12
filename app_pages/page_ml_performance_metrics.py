import streamlit as st
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Define Variables
file_path = ('outputs/v1')
labels_distribution_image = os.path.join(file_path, 'labels_distribution.png')
model_training_acc_image = os.path.join(file_path, 'model_training_acc.png')
model_training_losses_image = os.path.join(file_path, 'model_training_losses.png')
confusion_matrix_image = os.path.join(file_path, 'confusion_matrix.png')
roc_curve_image = os.path.join(file_path, 'roc_curve.png')



def page_ml_performance_metrics_body():
    """ Show Performance Metrics Images """

    st.write("### ML Performance Metrics")

    # Labels Distribution
    st.write("#### **Labels Distribution**")
    st.image(labels_distribution_image, caption="Labels Distribution")

    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        # Model Training Accuracy
        st.write("#### **Model Training Accuracy**")
        st.image(model_training_acc_image, caption="Model Training Accuracy")

    with col2:
        # Model Training Losses
        st.write("#### **Model Training Losses**")
        st.image(model_training_losses_image, caption="Model Training Losses")

    st.write("---")

    col1, col2 = st.columns(2)
    with col1:
        # Confusion Matrix
        st.write("#### **Confusion Matrix**")
        st.image(confusion_matrix_image, caption="Confusion Matrix")
    with col2:
        # ROC Curve
        st.write("#### **ROC Curve**")
        st.image(roc_curve_image, caption="ROC Curve")

    st.write("---")