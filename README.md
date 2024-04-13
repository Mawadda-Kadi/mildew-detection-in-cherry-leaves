#  Cherry Powdery Mildew Detector

**Here is the Livesite:**
[Cherry Powdery Mildew Detector]()
---

### Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Rationale for the Model](#rationale-for-the-model)
5. [Implementation of the Business Requirements](#implementation-of-the-business-requirements)
6. [The Rationale to Map the Business Requirements to the Data Visualizations and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
7. [ML Business Case](#ml-business-case)
8. [CRISP-DM Process](#crisp-dm-process)
9. [Dashboard Design](#dashboard-design)
10. [Manual Testing for Dashboard Pages](#manual-testing-for-dashboard-pages)
11. [Unfixed Bugs](#unfixed-bugs)
12. [Deployment](#deployment)
13. [Technologies Used](#technologies-used)
14. [Credits](#credits)

---

## Dataset Overview
* The dataset utilized in this project originates from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves) and forms the basis for a hypothetical scenario demonstrating how predictive analytics can be implemented effectively in real-world projects.
* Comprising over 4,000 images, the dataset features photographs of cherry leaves collected directly from the client's agricultural fields. These images include representations of both healthy leaves and those afflicted with powdery mildew—a common fungal disease that impacts a variety of plants. Given that cherry crops represent a significant aspect of the client's product lineup, there is a pressing concern regarding the potential delivery of products of diminished quality to the market due to this disease.

---

## Business Requirements
The cherry crop from Farmy & Foods is currently experiencing issues with powdery mildew infestation. The existing approach to manage this involves manual inspections where an employee examines each cherry tree for about 30 minutes, sampling leaves to determine their health status. If powdery mildew is detected, it takes an additional minute per tree to apply a fungicidal treatment. With thousands of cherry trees spread across various farms nationwide, this manual method is inefficient and not feasible at scale.

In an effort to streamline this process, the IT team has proposed the implementation of a machine learning (ML) system capable of instantly determining the health of a cherry leaf from an image. This system aims to replace the time-consuming manual inspections. This initiative, if successful, holds potential for adaptation to other crops within the company that require pest detection. The dataset used for this purpose includes a variety of cherry leaf images collected by Farmy & Foods from their orchards.


* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

---

## Hypothesis and Validation
Creating and validating hypotheses is a fundamental part of scientific research and data analysis, especially in fields like machine learning.

### 1. Hypothesis on Image Resolution
**Hypothesis**: Higher image resolution leads to better model accuracy in detecting powdery mildew in cherry leaves.

#### Introduction
Investigating the influence of image resolution on the detection of powdery mildew in cherry leaves, we hypothesize that varying resolutions affect the model's detail capture and, consequently, its accuracy.

#### How to Validate:
- **Set up distinct resolutions**: Define three categories for image resolution — low (64x64), medium (128x128), and high (256x256).
- **Prepare the data**: Use `ImageDataGenerator` to process images from the dataset into these three resolutions for training and testing the model.
- **Train separate models**: For each resolution category, train a model with the same architecture and hyperparameters to ensure consistency in comparison.
- **Evaluate performance**: Test each model on a separate validation set that is consistent across all resolutions to measure and compare accuracy, precision, recall, and F1 score.
- **Analyze results**: Document the performance metrics for each resolution to identify patterns or changes in model behavior as image detail varies.
- **Draw conclusions**: Based on the comparative analysis, conclude which resolution provides the optimal balance between detail capture and computational efficiency for detecting powdery mildew in cherry leaves.

#### Observation
Testing across different resolutions showed all provided high accuracy, with very slight improvements in medium and high settings.

Accuracy by resolution: {'low': 0.9952606558799744, 'medium': 0.9964454770088196, 'high': 0.9964454770088196}


#### Conclusion
The model effectively identifies powdery mildew at varying resolutions, indicating a good balance between detail recognition and computational efficiency. Excessively high resolution doesn't markedly improve detection, highlighting the model's ability to generalize from the key features of the condition.


### 2. Hypothesis on Data Augmentation
**Hypothesis**: Applying data augmentation techniques improves the model's ability to generalize and increases its accuracy in detecting powdery mildew.

#### Introduction
This hypothesis examines whether data augmentation enhances the performance of a model trained to identify powdery mildew in cherry leaves. Data augmentation artificially expands the training set with transformed versions of images, potentially improving the model's robustness and ability to generalize.

#### How to Validate
- **Generate augmented and non-augmented datasets**: Use `ImageDataGenerator` to create two sets of data from the training path, one with augmentation (rotation, shift, shear, zoom, flip) and one without.
- **Train models separately**: Build and train two identical models, one on each dataset, maintaining the same architecture and training conditions.
- **Evaluate performance**: Test both models on an untouched test set, comparing key metrics such as accuracy, precision, recall, and F1 score.
- **Analyze results**: Check if the augmented data model performs better in terms of accuracy, precision, recall, and F1 score.
- **Draw conclusions**: Assess whether data augmentation leads to a statistically significant improvement in model performance.

#### Observation
The model trained without data augmentation achieved an accuracy of 99.88%, precision of 100%, recall of 99.76%, and F1 score of 99.88%. In contrast, the model trained with augmentation showed an accuracy of 99.53%, precision of 100%, recall of 99.05%, and F1 score of 99.52%. Both models exhibited high performance, but the non-augmented model scored slightly higher in all metrics except precision, which remained perfect in both cases.

#### Conclusion
The slight decrease in performance metrics for the augmented model suggests that while data augmentation introduces more variability and robustness against overfitting, in this case, the original data set might already be sufficient or well-representative of the problem space, leading to high performance even without augmentation. Given the high precision in both models, false positives were effectively minimized. The augmented model showed a minor reduction in recall and F1 score, indicating a slightly lower ability to detect all positive cases.

---

## The Rationale to Map the Business Requirements to the Data Visualizations and ML Tasks
The business requirements for this project were decomposed into several user stories, which were then translated into specific Machine Learning tasks. All tasks were manually tested and function as expected, ensuring that the system is both robust and user-friendly.

**Business Requirement 1: Data Visualization**
The client wants to visually distinguish a cherry leaf affected by powdery mildew from a healthy one.

#### User Stories:
- As a client, I want to navigate easily around an interactive dashboard to view and understand the data presented.
- As a client, I want to display the "mean" and "standard deviation" images for cherry leaves that are healthy and those that contain powdery mildew, so that I can visually differentiate between the two.
- As a client, I want to display the difference between an average healthy cherry leaf and one that contains powdery mildew, to enhance my ability to visually distinguish them.
- As a client, I want to display an image montage for cherry leaves that are healthy and those that contain powdery mildew, to aid in visual differentiation.

#### Implementation:
- Developed a Streamlit-based dashboard with easy navigation and interactive visual elements (see Dashboard design for detailed presentation).
- Calculated and displayed the difference between average images of infected and healthy leaves.
- Generated "mean" and "standard deviation" images for both healthy and powdery mildew-infected leaves.
- Implemented an image montage feature for displaying a variety of healthy and infected leaves.

**Business Requirement 2: Classification**
The client needs to determine whether a given cherry leaf is affected by powdery mildew.

#### User Stories:
- As a client, I want a ML model to predict with an accuracy of at least 86% whether a given cherry leaf is healthy or contains powdery mildew.

#### Implementation:
- The rationale for the ML model deployed is detailed in the "Hypothesis and Validation" section of the dashboard.
- Enabled an uploader widget on the dashboard allowing clients to upload cherry leaf images in .jpeg format for instant evaluation.
- The system can handle multiple images up to 200MB at once, displaying each uploaded image along with a prediction statement indicating if the leaf is infected with powdery mildew and the associated probability.

#### Key Features:
- Interactive data visualizations to facilitate user engagement and understanding.
- Streamlined user interface on the dashboard for easy navigation and accessibility.
- Comprehensive display of predictive results alongside visual data aids.
- By addressing these user stories through carefully designed data visualization and machine learning tasks, this project not only meets the defined business requirements but also enhances the client's ability to make informed decisions based on robust data analysis and model predictions.

---

## ML Business Case

### Powdery Mildew Classifier for Cherry Leaves
The objective is to develop a machine learning model capable of predicting whether a cherry leaf is infected with powdery mildew based on image analysis. This problem is classified as supervised learning, specifically a two-class, single-label classification model. The ultimate goal is to provide a tool that can aid farmers and agricultural specialists in quickly and accurately detecting powdery mildew, a common and damaging plant disease.

**Ideal Outcome**:
The primary outcome is to equip farmers with a reliable tool for early detection of powdery mildew, facilitating timely and effective disease management.

**Model Success Metrics**:
- **Accuracy**: Achieve at least 97% accuracy on the test set.
- **Model Output**: The output is a binary flag coupled with a probability score, indicating the likelihood of powdery mildew presence on a leaf.

**Heuristics**:
- **Current Detection Method**: Traditionally, detection involves manual inspection of leaves by farmers, which is time-consuming (approximately 30 minutes per tree) and subject to human error. This method's limitations underscore the need for an automated solution.

**Data Source**:
- The model will be trained using a dataset provided by a collaborative agricultural project, consisting of 4208 images categorized into healthy and powdery mildew-infected cherry leaves.

**Approach**:
- **Data Preparation**: Utilize a TensorFlow-based CNN (Convolutional Neural Network) model with specified layers and hyperparameters designed to optimize performance for image classification tasks.
- **Model Design**:
  - **Convolutional Layers**: Capture spatial hierarchies in the images.
  - **Global Average Pooling**: Reduces model complexity and computational load.
  - **Regularization Techniques** (L2 regularization, Dropout): Prevent overfitting and enhance model generalization.
- **Training Strategy**:
  - **Learning Rate Scheduler**: Adjusts the learning rate based on training epochs to improve convergence.
  - **Early Stopping**: Monitors validation loss to halt training when improvement stalls, preserving the best model state.

#### Rationale Behind the Hyperparameter Choices in Data Modeling
1. **Filters in Convolutional Layers:**
   - **Configuration**: Started with 32 filters in the first layer and doubled to 64 in subsequent layers.
   - **Rationale**: Increasing the number of filters in deeper layers allows the model to capture more complex features in the images. Starting with fewer filters helps in extracting basic features and gradually increasing allows the model to build up a more detailed understanding of the images.

2. **Kernel Size:**
   - **Configuration**: 3x3 for all convolutional layers.
   - **Rationale**: A 3x3 kernel size is generally effective for capturing spatial hierarchies in image data. It strikes a good balance between capturing detail and computational efficiency.

3. **Activation Function:**
   - **Configuration**: ReLU (Rectified Linear Unit) for all hidden layers.
   - **Rationale**: ReLU helps in introducing non-linearity to the model, enabling it to learn more complex patterns. It is also computationally efficient, which is beneficial given the high-resolution images in our dataset.

4. **Dense Units:**
   - **Configuration**: 64 units in the dense layer preceding the output.
   - **Rationale**: Provides a sufficiently large, yet manageable number of neurons to process the features extracted by the convolutional layers, aiding in the final classification.

5. **Dropout Rate:**
   - **Configuration**: 0.5 in the dropout layer.
   - **Rationale**: Dropout is used to prevent overfitting by randomly dropping units during training. A rate of 0.5 is aggressive yet commonly used, helping to significantly reduce overfitting while allowing enough capacity for learning.

6. **Learning Rate and Scheduling:**
   - **Configuration**: Starts with a default learning rate (adaptive based on the optimizer) and decreases exponentially after 10 epochs.
   - **Rationale**: Starting with a higher learning rate allows the model to rapidly converge towards the general vicinity of the optimal solution. Reducing the learning rate as training progresses allows for finer adjustments, improving the model’s ability to reach the best possible solution.

**Model Deployment**:
- Users can upload images directly to an application interface, receiving immediate predictions.
- The model processes images on the fly, ensuring fast and efficient performance suitable for real-time applications.

This ML business case is structured to not only meet the immediate needs of the client by providing a high-accuracy diagnostic tool but also to advance the broader agricultural community's ability to manage plant health more effectively. The strategic use of machine learning here aims to replace slower, less reliable manual methods with a swift, scientific approach, thereby reducing the economic impact of plant diseases like powdery mildew on commercial crops.

---

## CRISP-DM Process
The CRISP-DM methodology provides a structured approach to planning a data mining project. It is often used as a standard approach in data science to ensure successful outcomes. Here's how the phases of CRISP-DM align with the activities in your project on cherry leaf disease detection:

### 1. Business Understanding
- **Goal Identification**: The primary objective is to develop a model that can accurately differentiate between healthy cherry leaves and those infected with powdery mildew.
- **Assessment of Current Situation**: The manual process of inspecting cherry leaves is time-consuming and prone to human error, necessitating an automated solution.

### 2. Data Understanding
- **Data Collection**: Images of cherry leaves, both healthy and infected, are sourced and organized for analysis.
- **Data Exploration**: Initial data exploration is conducted in `1-data-collection.ipynb` where data is prepared, cleaned, and split into training, validation, and test sets.

### 3. Data Preparation
- **Data Cleaning**: Removing non-image files and handling any anomalies or corrupt data.
- **Data Construction**: Creating training, validation, and test datasets with appropriate splits to ensure model generalization.
- **Data Integration**: Merging data from multiple sources, if necessary, to create a comprehensive dataset for training.

### 4. Modeling
- **Model Selection**: Choosing appropriate machine learning algorithms. In this case, a convolutional neural network (CNN) is tailored for image classification tasks.
- **Model Building**: Configuring the model with selected hyperparameters in `3-modelling-and-evaluating.ipynb`.
- **Model Training**: Conducting the training process using augmented data to enhance model robustness.

### 5. Evaluation
- **Model Assessment**: Evaluating the model's performance on a hold-out test set to estimate how well it might perform in general when used to make predictions on new data.
- **Performance Metrics**: Accuracy, precision, recall, F1 score, confusion matrix, and ROC curves are calculated to assess the quality of the model comprehensively.

### 6. Deployment
- **Deployment Planning**: Preparing for the deployment of the machine learning model in a production environment where it can be accessed by end-users, such as integration into a Streamlit dashboard.
- **Monitoring and Maintenance**: Establishing a plan for regular updates and maintenance based on new data or changing conditions in disease detection criteria.

### 7. Project Review
- **Review Process**: Evaluating the entire process for efficiency and effectiveness, identifying any areas for improvement in future iterations of the model or data collection.
- **Results Documentation**: Final documentation and reporting of the project outcomes, including insights gained and the potential business impact.

This structured approach ensures that every aspect of the machine learning project is thoroughly planned, executed, and reviewed, leading to a robust solution that meets the business requirements effectively.

---

## Dashboard Design

### **Page 1: Quick Project Summary**

![summary-page](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/assets/151715427/b55c08fd-d102-42ef-9e2c-db212673a1e4)

**Quick Project Summary**

**General Information**
Powdery mildew is a fungal disease affecting many plant species, characterized by white powdery spots on leaves and stems. Early detection is crucial to manage and control the spread of this disease effectively.
The project dataset contains high-resolution images of cherry leaves, categorized into healthy and powdery mildew-infected samples.
**Project Dataset**
The dataset consists of images categorized into two classes: healthy cherry leaves and leaves infected with powdery mildew. The detailed examination of these leaves allows for the development of a model capable of distinguishing between the two conditions.
**Business Requirements**
The client is interested in:
1. Visually differentiating between healthy leaves and those infected with powdery mildew.
2. Developing a method to accurately detect the presence of powdery mildew on cherry leaves.

### **Page 2: Leaves Visualizer**
![visualiser-page](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/assets/151715427/c04b446e-2d1f-4ef9-bb9f-1ef3c82e901f)

To address the first business requirement, the interface will feature:
- Checkbox 1: Display the average and variability images of healthy and infected leaves.
- Checkbox 2: Show differences between average images of healthy and infected leaves.
- Checkbox 3: Image montage to visualize a collection of leaves from each category.

### **Page 3: Powdery Mildew Detection**
![detector-page](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/assets/151715427/6960a2f4-1db5-407d-bf9e-1236aa317fd7)

For the second business requirement, create a user interface allowing users to upload cherry leaf images. The system will analyze each image and display:
- The image.
- A prediction statement indicating whether the leaf is infected with powdery mildew.
- The probability associated with the prediction.
- A table will list the image names and prediction results, with a download button to export this data.

### **Page 4: Project Hypothesis and Validation**
![hypothesis-page](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/assets/151715427/3a17443e-fde3-49c1-9279-4c035a132e0c)

Each project hypothesis will be outlined with a corresponding block detailing the conclusions and validation methods used to verify the hypothesis.

### **Page 5: ML Prediction Metrics**
![performance-page](https://github.com/Mawadda-Kadi/mildew-detection-in-cherry-leaves/assets/151715427/4327e413-fa44-47a6-a93d-264a45eb69cd)

This section will include:
- Label frequencies for the training, validation, and test sets.
- Model history, including accuracy and loss over epochs.
- Model evaluation results with metrics such as Confusion Matrix, and ROC curve.

This design document framework lays out how the project's objectives align with the client's needs, detailing the tools and methods used to achieve and validate these goals.

---

## Manual Testing for Dashboard Pages

### Quick Project Summary
- **Objective**: Verify that the page provides a concise overview of the project.
- **Test Steps**:
  1. Navigate to the "Quick Project Summary" page.
  2. Read through the provided information to ensure it accurately represents the project's purpose and dataset details.
  3. Check that the README link is clickable and direct to the correct page.

### Leaf Visualizer
- **Objective**: Ensure that the visualizer correctly displays images and differentiates between healthy and powdery mildew-infected leaves.
- **Test Steps**:
  1. Navigate to the "Leaf Visualizer" page.
  2. Interact with each checkbox to display images:
     - Average and variability images.
     - Differences between average images.
     - Image montage.
  3. Verify that the correct images load without errors and that the distinctions are clear and accurate.
  4. Check that the README link is clickable and direct to the correct page.

### Powdery Mildew Detection
- **Objective**: Test the functionality of the image upload and prediction feature.
- **Test Steps**:
  1. Navigate to the "Powdery Mildew Detection" page.
  2. Use the file uploader to upload a cherry leaf image.
  3. Confirm that the uploaded image displays correctly.
  4. Check the prediction output to see if it accurately identifies the leaf's health status.
  5. Verify that the probability score is displayed alongside the prediction.
  6. Check that the README link is clickable and direct to the correct page.

### Project Hypothesis and Validation
- **Objective**: Check the presentation of hypotheses and their validation.
- **Test Steps**:
  1. Navigate to the "Project Hypothesis and Validation" page.
  2. Review each hypothesis to ensure it is clearly stated and the validation steps are comprehensively detailed.
  3. Confirm that all related results and conclusions are logically presented and supported by data.
  4. Check that the README link is clickable and direct to the correct page.

### ML Performance Metrics
- **Objective**: Ensure that machine learning performance metrics are accurately reported and visualized.
- **Test Steps**:
  1. Navigate to the "ML Performance Metrics" page.
  2. Examine the displayed metrics such as accuracy, precision, recall, F1-score, and the ROC curve.
  3. Check that all visualizations (graphs and charts) render correctly and match the expected results from the model evaluation.
  4. Validate that historical training and validation losses and accuracies are correctly plotted over epochs.
  5. Check that the README link is clickable and direct to the correct page.

All the pages and features of the dashboard have been thoroughly manually tested and have successfully passed the tests.

---

## Unfixed Bug
- Prediction Inaccuracy: When uploading an image of a healthy cherry leaf, the system incorrectly predicts it as affected by powdery mildew. This issue may stem from model overfitting or insufficient representation of healthy leaves in the training data.

---

## Deployment
### Heroku
The steps needed to deploy this projects are as follows:
1. Create a requirement.txt file in GitHub, for Heroku to read, listing the dependencies the program needs in order to run.
2. Ensure that you select a version that is compatible with your application's requirements. Visit the [Heroku documentation](https://devcenter.heroku.com/articles/python-support).
3. push the recent changes to GitHub and go to your Heroku account page to create and deploy the app running the project.
4. Chose "CREATE NEW APP", give it a unique name, and select a geographical region.
5. Add heroku/python buildpack from the Settings tab.
6. From the Deploy tab, chose GitHub as deployment method, connect to GitHub and select the project's repository.
7. Select the branch you want to deploy, then click Deploy Branch.
8. Click to "Enable Automatic Deploys " or chose to "Deploy Branch" from the Manual Deploy section.
9. Wait for the logs to run while the dependencies are installed and the app is being built.
10. The mock terminal is then ready and accessible from a link similar to https://your-projects-name.herokuapp.com/
11. If the slug size is too large then add large files not required for the app to the .slugignore file.

### Fork the Repository
1. Navigate to the GitHub page of the repository you want to fork.
2. Click on the "Fork" button, usually located at the top right of the page.
3. This action creates a copy of the repository under your GitHub account.

### Clone the Repository
1. Go to your GitHub account, find the forked repository, and open it.
2. Click on the "Code" button, then choose "HTTPS" or "SSH" based on your preference and copy the URL provided.
3. Open your terminal or Git Bash.
4. Use the `git clone` command followed by the copied URL to clone the repository to your local machine. For example:
     ```
     git clone https://github.com/yourusername/repository-name.git
     ```
5. This command creates a local copy of the forked repository on your computer.

---

## Technologies Used
- **Python**: Used as the primary programming language for development, data manipulation, and machine learning.
- **GitHub**: Employed for version control and source code management.
- **Jupyter Notebook**: Utilized for interactive data collection, visualization, cleaning, as well as for model training and evaluation.
- **Streamlit**: Deployed for building interactive and user-friendly dashboards.
- **Heroku**: Chosen for hosting the application, allowing it to be accessed easily over the web.
- **Chat GPT**: Used for troubleshooting

### Main Data Analysis and Machine Learning Libraries
This project utilizes several powerful data analysis and machine learning libraries to process data, train models, and evaluate results. Below is a list of the primary libraries used, along with examples of how they were applied throughout the project:

### 1. **NumPy**
- **Purpose**: Used for numerical operations on arrays and matrices.
- **Example**: NumPy was used to convert images into arrays for model training and to perform transformations like normalization.

```python
import numpy as np
# Convert PIL image to numpy array
image_array = np.array(pil_image) / 255.0
```

### 2. **Pandas**
- **Purpose**: Provides data manipulation and analysis tools.
- **Example**: Pandas was employed to manage datasets, particularly for organizing and storing prediction results in a structured format.

```python
import pandas as pd
# Create a DataFrame to store prediction results
results_df = pd.DataFrame(data={'Image Name': image_names, 'Prediction': predictions})
```

### 3. **Matplotlib**
- **Purpose**: A plotting library for creating static, interactive, and animated visualizations in Python.
- **Example**: Used for plotting training and validation loss and accuracy graphs.

```python
import matplotlib.pyplot as plt
# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()
```

### 4. **TensorFlow and Keras**
- **Purpose**: TensorFlow is an end-to-end open-source platform for machine learning, and Keras is a high-level neural networks API running on top of TensorFlow.
- **Example**: TensorFlow and Keras were used to build and train the deep learning models, handle data augmentation, and perform model evaluations.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 5. **Scikit-Learn**
- **Purpose**: Offers simple and efficient tools for predictive data analysis.
- **Example**: Used for generating classification reports and confusion matrices to evaluate model performance.

```python
from sklearn.metrics import classification_report, confusion_matrix

# Calculate metrics
y_pred = model.predict_classes(test_set)
print(classification_report(test_set.classes, y_pred))
print(confusion_matrix(test_set.classes, y_pred))
```

### 6. **Streamlit**
- **Purpose**: Streamlit is an open-source app framework specifically for Machine Learning and Data Science teams to create beautiful, interactive web apps quickly and with minimal code.
- **Example**: In this project, Streamlit was used extensively to build an interactive dashboard that allows users to upload cherry leaf images, view model predictions, and navigate through different data visualizations.

```python
import streamlit as st

def page_project_summary_body():
    st.title("Cherry Leaves Powdery Mildew Detector")

```

### 7. **Seaborn**
- **Purpose**: Seaborn is a Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.
- **Example**: In this project, Seaborn was utilized to create enhanced visualizations for comparing the distribution of healthy versus powdery mildew-infected cherry leaves, which helps in understanding the data better and making informed decisions.

```python
import seaborn as sns

def plot_label_distribution(labels):
    sns.set_theme(style="whitegrid")
```

These libraries form the backbone of the project's analytical and machine learning operations, facilitating a wide range of tasks from data preprocessing and model building to evaluation and visualization.

---

## Credits

#### Educational Sources:
- [Code Institute Lessons](https://learn.codeinstitute.net/ci_program/diplomainsoftwaredevelopmentpredictiveanalytics)
- [Kaggle Documentation](https://www.kaggle.com/docs)
- [Matplotlib Tutorial](https://www.w3schools.com/python/matplotlib_intro.asp)
- [TensorFlow Core](https://www.tensorflow.org/guide)
- [Scikit Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [ResNet-50: The Basics and a Quick Tutorial](https://datagen.tech/guides/computer-vision/resnet-50/)
- [How to Build a Machine Learning Model](https://towardsdatascience.com/how-to-build-a-machine-learning-model-439ab8fb3fb1)
- [A Complete Guide to Data Augmentation](https://www.datacamp.com/tutorial/complete-guide-data-augmentation)
- [Machine Learning - Data Distribution](https://www.w3schools.com/python/python_ml_data_distribution.asp)
- [Machine Learning - Train/Test](https://www.w3schools.com/python/python_ml_train_test.asp)
- [Machine Learning - AUC - ROC Curve](https://www.w3schools.com/python/python_ml_auc_roc.asp)
- [Machine Learning - Confusion Matrix](https://www.w3schools.com/python/python_ml_confusion_matrix.asp)
- [Hypothesis in Machine Learning](https://www.geeksforgeeks.org/ml-understanding-hypothesis/)
- [CNN | Introduction to Padding](https://www.geeksforgeeks.org/cnn-introduction-to-padding/?ref=ml_lbp)
- [CNN | Introduction to Pooling Layer](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/?ref=lbp)
- [CIFAR-10 Image Classification in TensorFlow](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/?ref=lbp)
- [How to Convert Images to NumPy Arrays and Back](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/#:~:text=With%20Pillow%20installed%2C%20you%20can,of%20pixels%20as%20an%20image.)
- [Using Learning Rate Scheduler and Early Stopping with PyTorch](https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/)
- [F1 Score in Machine Learning: Intro & Calculation](https://www.v7labs.com/blog/f1-score-guide#:~:text=for%20Machine%20Learning-,What%20is%20F1%20score%3F,prediction%20across%20the%20entire%20dataset.)
- [How to Calculate Precision, Recall, F1, and More for Deep Learning Models](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)

#### Content:
- I implemented similar README file structure from [cla-cif/Cherry-Powdery-Mildew-Detector](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/README.md)
- Project Summary page Content from [Identification of Cherry Leaf Disease Infected by Podosphaera Pannosa via Convolutional Neural Network](https://www.researchgate.net/publication/331252859_Identification_of_Cherry_Leaf_Disease_Infected_by_Podosphaera_Pannosa_via_Convolutional_Neural_Network)

#### Media:
- Dataset from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves)
- Healthy Leaf Image in README file from [Creative Market](https://creativemarket.com/AnatoliySadovskiy/736161-Sweet-cherry-leaf)

#### Code:
I adapted code from [Code-Institute-Org/WalkthroughProject01](https://github.com/Code-Institute-Org/WalkthroughProject01) for the following:
- Data Collection and Preparation
- Image Contrast Analysis: Used for creating contrast plots to differentiate between healthy and powdery mildew-infected cherry leaves
- Image Montage Creation
- Data Augmentation
- Data Prediction: Adapted to implement model inference for evaluating individual images for signs of powdery mildew

### Acknowledgements
I would like to express my gratitude to my mentor, Mo Shami, for his guidance and support throughout this project.

---