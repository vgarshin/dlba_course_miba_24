#!/usr/bin/env python
# coding: utf-8

import os
import json
import streamlit as st


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


st.set_page_config(
    page_title='AI image assistant',
    page_icon=":mag:",
)

st.header('AI image assistant', divider='rainbow')
st.sidebar.success('Select an option to process your image')

st.markdown(
    """
    AI image assistant is an application that will help you
    to work with your images and organize them by categories.
    It can also help you to identify your friends at the photo
    or recognize and extract text from the images.
    ### The application can:
    - describe images
    - find objects and people
    - find the right category for image
    ### How does it work?
    - many computer vision models are under the hood
    - images are stored in a database with categories
    - application has a friendly UI to help you
    """
)
st.divider()

APP_CONFIG = read_json(file_path='config.json')
IMGS_PATH = APP_CONFIG['imgs_path']
DB_PATH = f'{IMGS_PATH}/db'

# create data and db folders
for path in [IMGS_PATH, DB_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

st.markdown(
    """
    More information:
    - DL course page https://github.com/vgarshin/dlba_course_miba_24
    - Telegram channel https://t.me/+9IWc1JocKT40MTJi
    """
)