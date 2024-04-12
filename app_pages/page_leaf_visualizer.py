import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
import itertools
import random

def page_leaf_visualizer_body():
    st.write("### Leaf Visualizer")
    st.info(
        "The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew."
    )

    st.info("""
    For additional information, please visit and **read** the
    [Project README file](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/blob/main/README.md)
    """)

    version = 'v1'
    if st.checkbox("Difference between average and variability image"):
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")

        st.warning(
            "While the average and variability images do not show stark differences, there might be subtle patterns that the model can learn."
        )

        st.image(avg_healthy, caption='Healthy Cherry Leaf - Average and Variability')
        st.image(avg_mildew, caption='Powdery Mildew Infected Cherry Leaf - Average and Variability')
        st.write("---")

    if st.checkbox("Differences between average healthy and average powdery mildew leaves"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff_healthy_powdery_mildew.png")

        st.warning("The differences between the average images of healthy and powdery mildew leaves might reveal subtle patterns.")
        st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"):
        st.write("To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = 'inputs/cherry_leaves_dataset/cherry-leaves'
        labels = os.listdir(os.path.join(my_data_dir, 'validation'))
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=os.path.join(my_data_dir, 'validation'),
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10,25))



def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))
        if nrows * ncols <= len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.error(f"Decrease nrows or ncols. There are {len(images_list)} images, requested montage with {nrows * ncols} spaces.")
            return

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for idx, (row, col) in enumerate(itertools.product(range(nrows), range(ncols))):
            img = imread(os.path.join(dir_path, label_to_display, img_idx[idx]))
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Width {img.shape[1]}px x Height {img.shape[0]}px")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error("The selected label does not exist in the dataset.")