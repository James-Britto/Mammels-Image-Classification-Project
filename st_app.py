import streamlit as st
import keras
import tensorflow as tf
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
model=tf.keras.models.load_model('./final_model/saved_model.h5')
st.title(':green[Mammal Classification App]')
uploaded_file=st.file_uploader(':blue[Upload File]')

folder_dict = {0: 'african_elephant',
               1: 'alpaca',
               2: 'american_bison',
               3: 'anteater',
               4: 'arctic_fox',
               5: 'armadillo',
               6: 'baboon',
               7: 'badger',
               8: 'blue_whale',
               9: 'brown_bear',
               10: 'camel',
               11: 'dolphin',
               12: 'giraffe',
               13: 'groundhog',
               14: 'highland_cattle',
               15: 'horse',
               16: 'jackal',
               17: 'kangaroo',
               18: 'koala',
               19: 'manatee',
               20: 'mongoose',
               21: 'mountain_goat',
               22: 'opossum',
               23: 'orangutan',
               24: 'otter',
               25: 'polar_bear',
               26: 'porcupine',
               27: 'red_panda',
               28: 'rhinoceros',
               29: 'seal',
               30: 'sea_lion',
               31: 'snow_leopard',
               32: 'squirrel',
               33: 'sugar_glider',
               34: 'tapir',
               35: 'vampire_bat',
               36: 'vicuna',
               37: 'walrus',
               38: 'warthog',
               39: 'water_buffalo',
               40: 'weasel',
               41: 'wildebeest',
               42: 'wombat',
               43: 'yak',
               44: 'zebra'}

if uploaded_file is None:
    st.write('<center>File yet to be uploaded',unsafe_allow_html=True)
else:
    b=uploaded_file.getvalue()
    img=Image.open(BytesIO(b))
    img=np.array(img)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_arr = np.expand_dims(img, axis=0)
    prediction=model.predict(im_arr)
    label=folder_dict[np.argmax(prediction)]
    st.write(f'The predicted label is: {label}')
    st.image(uploaded_file)

