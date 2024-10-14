#!/usr/bin/env python
# coding: utf-8

import os
import io
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

####################################
########## YOUR CODE HERE ##########
####################################
# You will need to download a model
# to implement summarization from 
# HugginFace Hub.
#
# You may want to use following models:
# https://huggingface.co/Falconsai/text_summarization
# https://huggingface.co/knkarthick/MEETING_SUMMARY
# ...or any other you like, but think of 
# the size of the model (<1GB recommended)
#
# Your code may look like this:
#from transformers import pipeline
#with st.spinner('Please wait, application is initializing...'):
#    MODEL_SUM_NAME = '<YOUR_MODEL>'
#    SUMMARIZATOR = pipeline("summarization", model=MODEL_SUM_NAME)
####################################

# page headers and info text
st.set_page_config(
    page_title='<YOUR_TEXT_HERE>', 
    page_icon=':microscope:'
)
st.sidebar.header('<YOUR_HEADER_HERE>')
st.header('<YOUR_HEADER_HERE>', divider='rainbow')

st.markdown(
    f"""
    <YOUR_DESCRIPTION_HERE>
    """
)
st.divider()

####################################
########## YOUR CODE HERE ##########
####################################
# Use example from the class with
# OCR model for text extracting from 
# the image or PDF file.
#
# Do not forget to add text summarization 
# model and display the output to the OCR 
# application's page  
####################################