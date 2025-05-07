#!/bin/bash
pip install -r requirements.txt
mkdir -p staticfiles
cp -r .streamlit staticfiles/
