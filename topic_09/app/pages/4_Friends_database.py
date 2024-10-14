#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Friends database', 
    page_icon=':microscope:'
)
st.sidebar.header('Friends database')
st.header('Database contains images of the friends', divider='rainbow')

st.markdown(
    f"""
    Here you can see all images of the friends 
    and update databes with new people.
    """
)
st.divider()

N_COLS = 3
APP_CONFIG = read_json(file_path='config.json')
IMGS_PATH = APP_CONFIG['imgs_path']
DB_PATH = f'{IMGS_PATH}/db'


@st.cache_data
def imgs_data(path):
    img_files = [
        f for f in os.listdir(path) 
        if f.lower().endswith('.jpg')
    ]
    data = [{
        'img_name': f.replace(f'{path}/', '').replace('.jpg', ''),
        'img_path': f'{path}/{f}'
    } for f in img_files]
    return data


# display a gallery of images
st.write('#### Gallery')
n_cols = st.slider('Width:', min_value=1, max_value=5, value=N_COLS)
imgs_list = imgs_data(path=DB_PATH)
cols = st.columns(n_cols)
for i, img in enumerate(imgs_list):
    with cols[i % n_cols]:
        st.image(
            img['img_path'], 
            caption=img['img_name'], 
            use_column_width=True
        )
st.divider()

# upload more images
st.write('#### Upload you image')
uploaded_file = st.file_uploader('Select an image file (JPEG format)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    if '.jpg' in file_name:
        bytes_data = uploaded_file.read()
        img = Image.open(io.BytesIO(bytes_data))
        img.save(f'{DB_PATH}/{file_name}')
        
        # logging
        msg = '{} - file "{}" saved in database\n'.format(
            datetime.datetime.now(),
            file_name
        )
        with open('history.log', 'a') as file:
            file.write(msg)
    else:
        st.error('File read error', icon='⚠️')
st.cache_data.clear()
st.divider()