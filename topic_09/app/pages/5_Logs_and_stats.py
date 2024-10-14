#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st

# page headers and info text
st.set_page_config(
    page_title='Logs and stats', 
    page_icon=':gear:'
)
st.sidebar.header(f'Logs for AI-assistant')
st.header('Stats of AI-assistant for image processing', divider='rainbow')

with open('history.log', 'r') as file:
    logs = file.read() 

st.text(logs)    
st.divider()
