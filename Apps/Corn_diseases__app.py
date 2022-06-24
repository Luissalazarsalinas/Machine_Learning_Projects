#!/usr/bin/env python
# coding: utf-8
## Import libraries 
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocess_input

#@st.cache
def get_model():
    model = tf.keras.models.load_model("Xceptio_model")
    return model

if __name__=='__main__':
    
    model = get_model()

    st.title('Corn Leaf Diseases Identifier')
    
    st.write('''
    Smart web application for identifying three diseases (Blight, common rust, grey leaf spot) infecting maize leaves. By a deep learning model.
           ''')
           
    ## Map class
    map_class = {
        0:'Blight',
        1:'Common Rust',
        2:'Gray Leaf Spot',
        3:'Healthy'
        }
    ## Dataframe 
    dict_class = {
        'Corn Leaf Diseases': ['Blight', 'Common Rust','Gray Leaf Spot','Healthy'],
        'Confiance': [0,0,0,0]
        }
        
    df_results = pd.DataFrame(dict_class, columns = ['Corn Leaf Diseases', 'Confiance'])
    
    def predictions(preds):
        
        df_results.loc[df_results['Corn Leaf Diseases'].index[0], 'Confiance'] = preds[0][0]
        df_results.loc[df_results['Corn Leaf Diseases'].index[1], 'Confiance'] = preds[0][1]
        df_results.loc[df_results['Corn Leaf Diseases'].index[2], 'Confiance'] = preds[0][2]
        df_results.loc[df_results['Corn Leaf Diseases'].index[3], 'Confiance'] = preds[0][3]

        return(df_results)

    ## load file
    uploaded_image = st.file_uploader('Choose an image file', type=['jpg','png','jpeg'])

    # upload images
    if not uploaded_image:
        st.write('Please upload an image before preceeding!')
        st.stop()
    else:
        # Decode the image and Predict the class
        img_as_bytes = uploaded_image.read() # Encoding image
        st.image(img_as_bytes, use_column_width= True) # Display the image
        img = tf.io.decode_image(img_as_bytes, channels = 3) # Convert image to tensor
        img = tf.image.resize(img,(224,224)) # Resize the image
        img_arr = tf.keras.preprocessing.image.img_to_array(img) # Convert image to array
        img_arr = tf.expand_dims(img_arr, 0) # Create a bacth


    img = xception_preprocess_input(img_arr)
    #img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction") 

      
    if Genrate_pred:
        st.title('Probabilities by Class') 
        preds = model.predict(img)
        preds_class = model.predict(img).argmax()

        st.dataframe(predictions(preds))
        st.title("The Corn Leaf is infected by {} disease".format(map_class[preds_class])) 
    

