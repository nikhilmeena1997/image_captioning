#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[3]:


model = load_model("model_weights/model_7.h5")


# In[6]:


from PIL import Image


# In[7]:


model_temp = ResNet50(weights = "imagenet" , input_shape=(224,224,3))


# In[8]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)

# In[9]:


def preprocess_img(img):
    img = Image.open(img)
    img=img.resize((224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img) #Normalisation 
    return img


# In[17]:


def encode_images(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector=feature_vector.reshape(1,feature_vector.shape[1])
    #print(feature_vector.shape)
    return feature_vector


# In[31]:


with open("./Storage/word_to_idx.pkl",'rb') as w2i:
    word_to_idx = pickle.load(w2i);
with open("./Storage/idx_to_word.pkl",'rb') as i2w:
    idx_to_word = pickle.load(i2w);


# In[39]:


def predict_caption(photo):
    
    input_text = "startseq"
    max_len=35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in input_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding = 'post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #Taking word with maximum probability
        word = idx_to_word[ypred]
        input_text += (' '+ word)
        
        if word == 'endseq':
            break
            
    final_caption = input_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[40]:
def caption_this_image(image):
    enc = encode_images(image)
    caption = predict_caption(enc)
    return caption

