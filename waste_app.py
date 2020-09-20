import tensorflow as tf 
import numpy as np 
import streamlit as st
from PIL import Image 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
# from tensorflow.keras.models import load_model
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
        # Waste Classification
        """
        )


st.write("This is a simple image classification web app to predict type of waste")
st.write("Sample Images of different waste types:")

cardboard = Image.open('sample_images/cardboard.jpg')
glass = Image.open('sample_images/glass.jpg')
metal = Image.open('sample_images/metal.jpg')
paper = Image.open('sample_images/paper.jpg')
plastic = Image.open('sample_images/plastic.jpg')
trash = Image.open('sample_images/trash.jpg')

st.image([cardboard,glass,metal,paper,plastic,trash],caption= ["cardboard","glass","metal","paper","plastic","trash"],width=150)

labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

@st.cache(allow_output_mutation =True)
def load_model():
    model = tf.keras.models.load_model('my_model.hdf5')
    return model


with  st.spinner('Loading Model....'):
    model = load_model()

def preprocess(img):
    img = image.array_to_img(img, scale=False)
    img = img.resize((256,256))
    img = image.img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img,axis=0)
    return img


file = st.file_uploader("Please upload waste image you want to classify", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded waste image yet")
else:
    original_img = Image.open(file)
    st.image(original_img,caption = 'Classifying Waste Image', use_column_width=True)
    img = image.img_to_array(original_img)
    a = preprocess(img)
    probs = model.predict(a)
    pred = labels[np.argmax(probs)]
    st.subheader("The Predicted Waste Type is : {wastetype}".format(wastetype = pred.capitalize()))
    st.write("Predicted Probabilities are: ")
    st.text("Index (0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash')")
    st.write(probs)
