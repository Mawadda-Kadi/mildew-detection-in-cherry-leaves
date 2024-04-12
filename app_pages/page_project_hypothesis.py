import streamlit as st

def page_project_hypothesis_body():
    st.title("Project Hypothesis and Validation")

    # Hypothesis 1
    st.header("1. Hypothesis on Image Resolution")
    st.write("""
    **Hypothesis**: Higher image resolution leads to better model accuracy in detecting powdery mildew in cherry leaves.

    **Introduction**
    Investigating the influence of image resolution on the detection of powdery mildew in cherry leaves,
    we hypothesize that varying resolutions affect the model's detail capture and, consequently, its accuracy.

    **How to Validate**
    - Define three categories for image resolution — low (64x64), medium (128x128), and high (256x256).
    - Use `ImageDataGenerator` to process images from the dataset into these three resolutions for training and testing.
    - Train a model with the same architecture and hyperparameters for each resolution category.
    - Evaluate performance using accuracy, precision, recall, and F1 score.
    - Analyze results to determine which resolution optimizes model performance.

    **Observation**
    Testing across different resolutions showed all provided high accuracy, with very slight improvements in medium and high settings.
    """)

    # Display accuracy by resolution
    resolution_data = {'low': 0.9953, 'medium': 0.9964, 'high': 0.9964}
    st.json(resolution_data)

    st.write("""
    **Conclusion**
    The model effectively identifies powdery mildew at varying resolutions, indicating a good balance between detail recognition and computational efficiency.
    """)

    st.write("---")

    # Hypothesis 2
    st.header("2. Hypothesis on Data Augmentation")
    st.write("""
    **Hypothesis**: Applying data augmentation techniques improves the model's ability to generalize and increases its accuracy.

    **Introduction**
    This hypothesis examines whether data augmentation enhances the performance of a model trained to identify powdery mildew in cherry leaves.

    **How to Validate**
    - Create two sets of data, one with augmentation and one without.
    - Train two identical models separately on these datasets.
    - Evaluate and compare their performance on an untouched test set.

    **Observation**
    The augmented model did not significantly outperform the non-augmented model, showing only minor variations in metrics.
    """)

    # Display the observed metrics
    metrics_data = {
        'Non-augmented': {'Accuracy': 0.9988, 'Precision': 1.0, 'Recall': 0.9976, 'F1 Score': 0.9988},
        'Augmented': {'Accuracy': 0.9953, 'Precision': 1.0, 'Recall': 0.9905, 'F1 Score': 0.9952}
    }

    st.json(metrics_data)

    st.write("""
    **Conclusion**
    - While data augmentation introduces more variability, the original dataset's representativeness might suffice, as indicated by high performance metrics.
    - Excessively high resolution doesn't markedly improve detection, highlighting the model's ability to generalize from the key features of the condition.
    """)

