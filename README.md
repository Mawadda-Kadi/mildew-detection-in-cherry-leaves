![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Codeanywhere Template Instructions

Welcome,

This is the Code Institute student template for Codeanywhere. We have preinstalled all of the tools you need to get started. It's perfectly ok to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Codeanywhere Template Instructions section of this README.md file,  and modify the remaining paragraphs for your own project. Please do read the Codeanywhere Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use.

## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into <a href="https://app.codeanywhere.com/" target="_blank" rel="noreferrer">CodeAnywhere</a> with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and <code>pip3 install -r requirements.txt</code>

1. In the terminal type <code>pip3 install jupyter</code>

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.

1. Open port 8888 preview or browser

1. Open the jupyter_notebooks directory in the jupyter webpage that has opened and click on the notebook you want to open.

1. Click the button Not Trusted and choose Trust.

Note that the kernel says Python 3. It inherits from the workspace so it will be Python-3.8.12 as installed by our template. To confirm this you can use <code>! python --version</code> in a notebook code cell.


## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.


## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.



## Business Requirements
The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.


* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


## Hypothesis and how to validate?
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






## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design

**Page 1: Quick Project Summary**

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

---

**Page 2: Leaves Visualizer**

To address the first business requirement, the interface will feature:
- Checkbox 1: Display the average and variability images of healthy and infected leaves.
- Checkbox 2: Show differences between average images of healthy and infected leaves.
- Checkbox 3: Image montage to visualize a collection of leaves from each category.

---

**Page 3: Powdery Mildew Detection**

For the second business requirement, create a user interface allowing users to upload cherry leaf images. The system will analyze each image and display:
- The image.
- A prediction statement indicating whether the leaf is infected with powdery mildew.
- The probability associated with the prediction.

A table will list the image names and prediction results, with a download button to export this data.

---

**Page 4: Project Hypothesis and Validation**

Each project hypothesis will be outlined with a corresponding block detailing the conclusions and validation methods used to verify the hypothesis.

---

**Page 5: ML Prediction Metrics**

This section will include:
- Label frequencies for the training, validation, and test sets.
- Model history, including accuracy and loss over epochs.
- Model evaluation results with metrics such as precision_score, recall_score, f1_score.

---

This design document framework lays out how the project's objectives align with the client's needs, detailing the tools and methods used to achieve and validate these goals.


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

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


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.


## Credits

* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.



## Acknowledgements (optional)
* Thank the people that provided support throughout this project.
