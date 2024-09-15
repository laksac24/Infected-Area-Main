import cv2
import numpy as np
from functions import YOLO_Pred
from flask import Flask, render_template, request
from functions2 import YOLO_Pred2
import pandas as pd
import base64


app=Flask(__name__)

yolo=YOLO_Pred('./Model2/weights/best.onnx','data.yaml')
yolo2=YOLO_Pred2('./Model2/weights/best.onnx','data.yaml')

@app.route("/")
def index():
    return render_template("index4.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files.get('image')
        print(image)
        if image:
            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img_pred = yolo.predictions(img)
            _, img_encoded = cv2.imencode('.jpg', img_pred)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            return render_template("index4.html", output_image=img_base64)
    return render_template("index4.html")

@app.route("/predict2",methods=['GET','POST'])
def predict2():
    result=yolo2.predictions2()
    def output2(main_img2):
        cv2.imshow('Plant_Prediction', main_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return(render_template("index4.html",img2=output2(result)))


if __name__=="__main__":
    app.run(debug=True)

# img=cv2.imread('./img1.JPG')
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img_pred=yolo.predictions(img)
# cv2.imshow('predictions',img_pred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()