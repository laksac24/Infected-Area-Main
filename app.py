import cv2
from functions import YOLO_Pred
from flask import Flask,render_template,request
from functions2 import YOLO_Pred2
import pandas as pd

app=Flask(__name__)

yolo=YOLO_Pred('C:/Users/DELL/OneDrive/Desktop/plantDP2.0/data_preparation/predictions/Model2/weights/best.onnx','data.yaml')
yolo2=YOLO_Pred2('C:/Users/DELL/OneDrive/Desktop/plantDP2.0/data_preparation/predictions/Model2/weights/best.onnx','data.yaml')

@app.route("/")
def index():
    return render_template("index4.html")

@app.route("/predict",methods=['GET','POST'])
def predict():
    image=request.form.get('image')
    img = cv2.imread(image)
    img_pred = yolo.predictions(img)
    def output(main_img):

        cv2.imshow('predictions',main_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return render_template("index4.html",output_image=output(img_pred))

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