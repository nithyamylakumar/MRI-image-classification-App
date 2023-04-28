import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import keras
import tensorflow as tf
import scipy
import os
from tqdm import tqdm
from skimage import io
import random
import glob
from keras.applications import imagenet_utils
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, normalize
from IPython.display import display
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator, array_to_img
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Conv2D,Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import imutils 
import random

##################################################################################################################################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

# Define a custom CSS style for the title with a different color
custom_style = """
<style>
.font_title {
    font-size: 50px;
    font-family: 'times';
    text-align: center;
    color: black; /* Change the color here */
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)
st.markdown("<p class='font_title'>Brain Tumor Multiclass Classification</p>", unsafe_allow_html=True)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/concept-brainstorming-artificial-intelligence-with-blue-color-human-brain-background_121658-753.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# #animation
# with open ("82546-brain.json") as source:
#     animation = json.load(source)
#     st_lottie(animation,width = 100, height=50)

st.subheader('There are four different types of brain tumors that can be classified using this app based on MRI scans')


# Add tab with dataset description
tab1, tab2, tab3, tab4= st.tabs(['About the Data', 'Image pre-processing', 'Tumors', 'Classification'])

with tab1:
    st.header('Dataset description')
    st.write('The dataset was obtained from Kaggle, and it consists of 7023 human brain MRI images. These images are categorized into four classes: Glioma, Meningioma, Pituitary, and Notumor.')
    st.write(" Efficient B1 ImageNet is a convolutional neural network (CNN) architecture that has shown excellent performance in various image classification tasks.This architecture has been used to classify the tumor images.")
with tab3:
    st.subheader('All about Tumors')
    tumor_selection = st.selectbox("Select a tumor:", ["Glioma", "Meningioma", "Pituitary", "Notumor"])

    if tumor_selection == "Glioma":
        st.write('Glioma: Glioma is a type of tumor that originates from glial cells, which are the supportive cells that make up the central nervous system (CNS), including the brain and spinal cord. Gliomas can be classified into different types based on the specific type of glial cell they arise from, such as astrocytoma, oligodendroglioma, ependymoma, and mixed gliomas. Gliomas are classified as either low-grade (grade I or II) or high-grade (grade III or IV) based on their aggressiveness and rate of growth.')
        st.image("images/glioma.jpg", caption="MRI image of glioma tumor", width = 600)
    elif tumor_selection == "Meningioma":
        st.write('Meningioma: Meningioma is a type of tumor that arises from the meninges, which are the protective membranes that cover the brain and spinal cord. Meningiomas are usually benign (non-cancerous) and slow-growing, but they can still cause symptoms depending on their size and location. Common symptoms of meningioma can include headaches, seizures, vision problems, memory loss, and changes in behavior or personality. Treatment options for meningioma may include observation, surgery, radiation therapy, or other targeted therapies depending on the specific characteristics and location of the tumor.')
        st.image("images/meningioma.jpg", caption="MRI image of meningioma tumor", width = 600)
    elif tumor_selection == "Pituitary":

        st.write("Pituitary tumors: Pituitary tumors are growths that occur in the pituitary gland, which is a small gland located at the base of the brain that plays a critical role in regulating various hormones in the body. Pituitary tumors can be classified as either benign (non-cancerous) or malignant (cancerous). Depending on their size and location, they can cause hormonal imbalances, visual disturbances, headaches, and other symptoms. Treatment options for pituitary tumors may include medication, surgery, radiation therapy, or a combination of these approaches, depending on the type, size, and location of the tumor, as well as the patient's overall health.")
        st.image("images/pituitary.webp", caption="MRI image of pituitary tumor",width = 600)
    elif tumor_selection == "Notumor":
        st.write('No-tumor: "No-tumor" generally refers to the absence of any abnormal growth or mass in a particular area of the body. It may indicate that the region being examined, such as the brain or other organs, does not show any evidence of tumors or abnormal tissue growth based on medical imaging or other diagnostic tests.')
        st.image("images/no-tumor.jpg", caption="MRI image of no tumor",  width = 600)
with tab2:
    st.header('Exploratory Data Analysis')
    st.write('Select from the dropdown to visualize the whole data and the EDA done on them')

    # Define a dictionary of images and their names
    image_dict = {"Glimpse of the data": "images/p3.png",
                "Cropped image": "images/c1.png",
                "Augmented image": "images/p2.png",
                "MRI Image with a mask": "images/p5.png"}

    # Create a selectbox for the user to choose which image to display
    selected_image = st.selectbox("Select an image:", list(image_dict.keys()))

    # Load the selected image using PIL
    image = Image.open(image_dict[selected_image])

    # Specify the width of the image display
    image_width = 700  # Set the desired width of the image in pixels

    # Display the selected image with the specified width
    st.image(image, caption=selected_image, width=image_width)
    

    #########################################################################################
with tab4:

    # Load the pre-trained model
    model = load_model('model_latest.h5')

    # Define class names
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    # Select a random test image file
    test_folder = 'test-data-app'  # Update with the path to your test folder
    test_image_files = os.listdir(test_folder)
    st.write("The saved classification model with the best validation accuracy has been used here for prediction on the test images. Although it classifies most of the images accurately, there are a few misclassifications. This has been kept for future work which involves hyperparameter tuning")
    selected_file = st.selectbox('Select a test image', test_image_files)
    selected_image_path = os.path.join(test_folder, selected_file)

    # Load and preprocess the selected image
    selected_image = Image.open(selected_image_path)
    selected_image_array = img_to_array(selected_image)
    selected_image_array = np.expand_dims(selected_image_array, axis=0)
    selected_image_array = imagenet_utils.preprocess_input(selected_image_array)

    # Make predictions
    predictions = model.predict(selected_image_array)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    # Display the selected image
    st.image(selected_image, caption='Selected Image', width = 400)

    # Display predicted and actual class names
    st.write('Predicted Class: ', predicted_class_name)
    st.write('Actual Class: ', selected_file.split('_')[0])  # Assumes image file names are in the format 'class_number.jpg'

    st.header('Test data results - Confusion matrix')
    st.image("images/cm.png", caption="Confusion matrix of the test data",  width = 600)
    st.write('The confusion matrix represents the number of true positive (TP), false positive (FP), true negative (TN), and false negative (FN) predictions for each class. In this case, the confusion matrix is a 4x4 matrix representing four classes: glioma, meningioma, no-tumor, and pituitary. The rows represent the actual classes, and the columns represent the predicted classes.For example, the number 57 in the first row and first column indicates that 57 glioma samples were correctly classified as glioma (true positives). Similarly, the number 122 in the first row and second column indicates that 122 glioma samples were incorrectly classified as meningioma (false positives). The number 75 in the first row and third column indicates that 75 glioma samples were incorrectly classified as no-tumor (false positives), and the number 46 in the first row and fourth column indicates that 46 glioma samples were incorrectly classified as pituitary (false positives).The same logic applies to the other three rows and columns, where we can see the number of correct and incorrect predictions for each class.')



