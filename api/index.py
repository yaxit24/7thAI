from flask import Flask, redirect, Response
import os
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def home():
    # Redirect to Streamlit Cloud hosting URL if you have one
    return 'Coursera Study Buddy API running. This is a backend API endpoint.'

@app.route('/<path:path>')
def catch_all(path):
    return f'Path: {path} not found. Please use the main app URL.'

# Make the app available to Vercel
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
