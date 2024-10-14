#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import streamlit as st
from PIL import Image


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Gallery', 
    page_icon=':microscope:'
)
st.sidebar.header('Images gallery')
st.header('Database contains images by the categories', divider='rainbow')

st.markdown(
    f"""
    Here you can see all the images
    classified by the categories with 
    help of AI-assistant.
    """
)
st.divider()

N_COLS = 3
APP_CONFIG = read_json(file_path='config.json')
IMGS_PATH = APP_CONFIG['imgs_path']
CLASSES = APP_CONFIG['classes']
DB_DICT = APP_CONFIG['db_dict']
CLASSES.append('other')
DB_DICT['other'] = 'other'


@st.cache_data
def imgs_data(path, classes, db_dict):
    data = {}
    for c in classes:
        img_files = [
            f for f in os.listdir(f'{path}/{db_dict[c]}') 
            if f.lower().endswith('.jpg')
        ]
        data_c = [{
            'img_name': f.replace(f'{path}/{db_dict[c]}', ''),
            'img_path': f'{path}/{db_dict[c]}/{f}'
        } for f in img_files]
        data[c] = data_c
    return data


# display a gallery of images
n_cols = st.slider('Width:', min_value=1, max_value=5, value=N_COLS)
imgs_list = imgs_data(path=IMGS_PATH, classes=CLASSES, db_dict=DB_DICT)
for c in CLASSES:
    st.write(f'#### Gallery of images from category -{DB_DICT[c]}-')
    cols = st.columns(n_cols)
    for i, img in enumerate(imgs_list[c]):
        with cols[i % n_cols]:
            st.image(
                img['img_path'], 
                caption=img['img_name'], 
                use_column_width=True
            )
    st.divider()
