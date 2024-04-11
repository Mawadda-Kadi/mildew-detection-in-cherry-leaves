import streamlit as st
from app_pages.multipage import MultiPage


# load pages scripts
from app_pages.page_project_summary import page_project_summary_body
from app_pages.page_leaf_visualizer import page_leaf_visualizer_body
from app_pages.page_powdery_mildew_detector import page_powdery_mildew_detector_body
#from app_pages.page_project_hypothesis import page_project_hypothesis_body
#from app_pages.page_ml_performance import page_ml_performance_metrics

# Create an instance of the app
app = MultiPage(app_name="Cherry Leaves Powdery Mildew Detector")

# Add app pages here using .add_page()
app.add_page("Quick Project Summary", page_project_summary_body)
app.add_page("Leaf Visualizer", page_leaf_visualizer_body)
app.add_page("Powdery Mildew Detection", page_powdery_mildew_detector_body)
#app.add_page("Project Hypothesis", page_project_hypothesis_body)
#app.add_page("ML Performance Metrics", page_ml_performance_metrics)

# Run the app
app.run()