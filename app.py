import cv2
import numpy as np
from functions import YOLO_Pred
from flask import Flask, render_template, request, flash
import pandas as pd
import base64
import logging
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
logging.basicConfig(level=logging.INFO)

yolo = YOLO_Pred('./Model2/weights/best.onnx', 'data.yaml')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image):
    try:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img_pred = yolo.predictions(img)
        _, img_encoded = cv2.imencode('.jpg', img_pred)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return img_base64
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return render_template("index.html")
        
        image = request.files['image']
        
        if image.filename == '':
            flash('No selected file')
            return render_template("index.html")
        
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            logging.info(f"Processing file: {filename}")
            img_base64 = process_image(image)
            if img_base64:
                return render_template("index.html", output_image=img_base64)
            else:
                flash('Error processing image')
        else:
            flash('Invalid file type')
    
    return render_template("index7.html")

if __name__ == "__main__":
    app.run(debug=True)