import streamlit as st


def page_project_summary_body():
    """
    Function to display the project summary page content.
    This function will be called in the main app script.
    """

    st.write("### Project Summary: Cherry Leaf Disease Detection")

    st.info("""
    **General Information**\n
    Powdery mildew is a fungal disease affecting many plant species,
    characterized by white powdery spots on leaves and stems.
    Early detection is crucial to manage and control the spread of
    this disease effectively.
    """)


    st.write("#### Introduction")
    st.write("""
    This project aims to detect and classify cherry leaf diseases,
    focusing on distinguishing between healthy leaves and those affected
    by powdery mildew.\n
    Cherry leaves affected by powdery mildew exhibit distinct patterns
    that can be captured through imaging.
    By using machine learning techniques,
    these patterns can be analyzed to determine the presence of disease.
    """)


    st.write("#### Description")
    st.write("""
    The dataset consists of images of cherry leaves, categorized into healthy and powdery mildew-affected classes.
    These images are used to train and validate the performance of the predictive model.
    """)

    st.info("""
    For additional information, please visit and **read** the
    [Project README file](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/blob/main/README.md)
    """)


    st.write("### The project has 2 business requirements:\n")
    st.success("""
    The client is interested in:
    1. Visually differentiating between healthy leaves and those infected with powdery mildew.
    2. Developing a method to accurately detect the presence of powdery mildew on cherry leaves.
    """)


