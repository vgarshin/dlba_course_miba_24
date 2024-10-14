#!/usr/bin/env python
# coding: utf-8

import os
import io
import cv2
import json
import datetime
import streamlit as st
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration
)
from ultralytics import YOLO
from deepface import DeepFace
from transformers import pipeline
import pandas as pd
import numpy as np


def img_caption(model, processor, img, text=None):
    """
    Uses BLIP model to caption image.
    
    """
    res = None
    if text:
        # conditional image captioning
        inputs = processor(img, text, return_tensors='pt')
    else:
        # unconditional image captioning
        inputs = processor(img, return_tensors='pt')
    out = model.generate(**inputs)
    res = processor.decode(out[0], skip_special_tokens=True)
    return res


def img_detect(model, img, plot=False):
    """
    Run YOLO inference on an image.
    
    """
    result = model(img)[0]
    boxes = result.boxes  # boxes object for bounding box outputs
    names = model.names
    objs = []
    for c, p in zip(boxes.cls, boxes.conf):
        objs.append({names[int(c)]: p.item()})
    img_bgr = result.plot()  # BGR-order numpy array
    img_rgb = Image.fromarray(img_bgr[..., ::-1])  # RGB-order PIL image
    if plot:
        plt.figure(figsize=(16, 8))
        plt.imshow(img_rgb)
        plt.show()
    return objs, img_rgb


def zeroshot(classifier, classes, img):
    scores = classifier(
        img,
        candidate_labels=classes
    )
    return scores


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Classify images', 
    page_icon=':bar_chart:'
)
st.sidebar.header('Classify images')
st.header('AI-assistant for your images', divider='rainbow')
st.markdown(
    f"""
    You can upload your image here and it will be 
    classified by predefined classes and saved to
    gallery.
    The assistant can also give a caption to the image
    and find your friends at the photo.
    """
)
st.divider()

# uploading models
with st.spinner('Please wait, application is initializing...'):
    
    # caption model
    MODEL_CAP_NAME = 'Salesforce/blip-image-captioning-base'
    PROCESSOR_CAP = BlipProcessor.from_pretrained(MODEL_CAP_NAME)
    MODEL_CAP = BlipForConditionalGeneration.from_pretrained(MODEL_CAP_NAME)

    # detection objects model
    MODEL_DET_NAME = 'yolov8n.pt'
    MODEL_DET = YOLO(MODEL_DET_NAME)
    ####################################
    ########## YOUR CODE HERE ##########
    ####################################
    # Experiment with new version YOLO
    # models and try segmentation model
    # as an alternative:
    # https://docs.ultralytics.com/tasks/segment/
    # NOTE: version 11 of YOLO available
    # Your code may look like this:
    #MODEL_SEG_NAME = '<YOUR_VERSION_YOLO>'
    #MODEL_SEG = YOLO(MODEL_SEG_NAME)
    ####################################
    
    # classification model and classes
    MODEL_ZERO_NAME = 'openai/clip-vit-base-patch16'
    CLASSIFIER_ZERO = pipeline('zero-shot-image-classification', model=MODEL_ZERO_NAME)
    ####################################
    ########## YOUR CODE HERE ##########
    ####################################
    # You may want to change the list of
    # classes so you will have to edit
    # `config.yaml` file and update it
    # with new data.
    # You do not have to change the code
    # below but think of naming classes
    # and categories for zero-shot classifier
    # to work correctly.
    # You will have to test your new classes
    # with ne images uploading
    ####################################
    APP_CONFIG = read_json(file_path='config.json')
    CLASSES = APP_CONFIG['classes']
    DB_DICT = APP_CONFIG['db_dict']
    TH_OTHERS = APP_CONFIG['th_others']
    IMGS_PATH = APP_CONFIG['imgs_path']
    for k, v in DB_DICT.items():
        imgs_path = f'{IMGS_PATH}/{v.strip()}'
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)
    imgs_path = f'{IMGS_PATH}/other'
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
            
    # faces detection  and recognition config
    DEEPFACE_MODELS = [
        'VGG-Face',
        'Facenet',
        'Facenet512',
        'OpenFace',
        'DeepFace',
        'DeepID',
        'ArcFace',
        'Dlib',
        'SFace',
        'GhostFaceNet'
    ]
    DB_PATH = '/home/jovyan/dlba/topic_09/app/data/db'
    
st.write('#### Upload you image')
uploaded_file = st.file_uploader('Select an image file (JPEG format)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    if '.jpg' in file_name:
        # input text for conditional image captioning
        text = st.text_input(
            'Input text for conditional image captioning (if needed)', 
            ''
        )
        with st.spinner('Please wait...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))
            
            # image caption model for uploaded image
            caption = img_caption(
                model=MODEL_CAP, 
                processor=PROCESSOR_CAP, 
                img=img, 
                text=text
            )
            st.write('##### Your image uploaded')
            st.image(img, caption=caption)
            st.divider()
            # logging
            msg = '{} - file "{}" got caption "{}"\n'.format(
                datetime.datetime.now(),
                file_name,
                caption
            )
            with open('history.log', 'a') as file:
                file.write(msg)
            
            # object detection model for uploaded image
            objs, img_det = img_detect(
                model=MODEL_DET, 
                img=img
            )
            objs = [list(x.keys())[0] for x in objs 
                    if x[list(x.keys())[0]] >= .5]
            objs = ' ,'.join(objs)
            st.write('##### Objects detected')
            st.image(img_det, caption='Object detection results', width=800)
            st.divider()
            st.caption('Objects found:')
            st.write(objs)
            st.divider()
            # logging
            msg = '{} - file "{}" objects detected "{}"\n'.format(
                datetime.datetime.now(),
                file_name,
                objs
            )
            with open('history.log', 'a') as file:
                file.write(msg)
            
            # classifying image with zero-shot modele
            scores = zeroshot(
                classifier=CLASSIFIER_ZERO, 
                classes=CLASSES, 
                img=img
            )
            max_score = sorted(scores, key=lambda x: x['score'])[-1]
            if max_score['score'] >= TH_OTHERS:
                category = max_score['label']
                save_path = DB_DICT[category]
            else:
                category = 'a photo of unknown stuff'
                save_path = 'other'
            img.save(f'{IMGS_PATH}/{save_path}/{file_name}')
            st.write('##### Classification results')
            st.image(img, caption=f'Looks like it is {category}')
            st.write(f'Image saved as: {save_path}')
            st.divider()
            # logging
            msg = '{} - file "{}" saved as "{}" category "{}"\n'.format(
                datetime.datetime.now(),
                file_name,
                save_path,
                category
            )
            with open('history.log', 'a') as file:
                file.write(msg)
            
            # plot a diagram with scores and scores output
            df = pd.DataFrame(scores)
            df = df.set_index('label')
            st.bar_chart(df)
            st.divider()
            
            # faces detection and recognition
            results = DeepFace.find(
                img_path=np.array(img),  # face to find
                db_path=f'{DB_PATH}',  # path to directory with faces
                model_name=DEEPFACE_MODELS[0],
                enforce_detection=False
            )
            st.write('##### Faces recognition results')
            found = []
            for result in results:
                name = result.identity.values
                if name:
                    found.append(
                        name[0].replace(f'{DB_PATH}/', '').replace('.jpg', '')
                    )
            if found:
                st.write(f'Found: {" ,".join(found)}')
            else:
                st.write('No known faces found')
            # logging
            msg = '{} - file "{}" {}\n'.format(
                datetime.datetime.now(),
                file_name,
                'faces found' + ' ,'.join(found) if found else 'no known faces found'
            )
            with open('history.log', 'a') as file:
                file.write(msg)
            
            ####################################
            ########## YOUR CODE HERE ##########
            ####################################
            # Your goal is to implement emotion 
            # recognition for the faces found.
            #
            # Use DeepFace framework as we did
            # in the previous classes. Print out 
            # the emotion for the people found 
            # at the image.
            # 
            # Your code may look like this:
            #results = DeepFace.analyze(<YOUR_IMG>)
            #for i, res in enumerate(results):
            #    emotion = res['dominant_emotion']
            ####################################
            
    else:
        st.error('File read error', icon='⚠️')
