import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
import uuid
import time

class YOLO_Pred2():
    def __init__(self,onnx_model,data_yaml):
        
        with open('data.yaml',mode='r') as f:
            data_yaml=yaml.load(f,Loader=SafeLoader)

        self.labels=data_yaml['names']
        
        self.yolo=cv2.dnn.readNetFromONNX('./Model2/weights/best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def predictions2(self):
        labels = ['leaves']
        number_imgs = 1
        IMAGES_PATH = os.path.join('trial_images')
        
        if not os.path.exists(IMAGES_PATH):
            if os.name == 'posix':
                os.makedirs(IMAGES_PATH, exist_ok=True)
                
            if os.name == 'nt':
                os.makedirs(IMAGES_PATH, exist_ok=True)
        for label in labels:
            path = os.path.join(IMAGES_PATH, label)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                
        for label in labels:
            cap = cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            for imgnum in range(number_imgs):
                print('Collecting image {}'.format(imgnum))
                ret, frame = cap.read()
                imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                time.sleep(3)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        folder_path = './trial_images/leaves'

        # List all files in the folder
        files = os.listdir(folder_path)

        # Loop through each file in the folder
        for index,file_name in enumerate(files):
            # Check if the file is an image (e.g., by extension)
            if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Add other extensions if needed
                # Construct the full file path
                old_file_path = os.path.join(folder_path, file_name)
                
                # Define a new file name, e.g., 'image_1.jpg', 'image_2.png', etc.
                new_file_name = "img1.jpg"
                new_file_path = os.path.join(folder_path, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
        img=cv2.imread('./trial_images/leaves/img1.jpg')
        image=img.copy()
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        row,col,d=image.shape
        
        max_rc=max(row,col)
        input_image=np.zeros((max_rc,max_rc,3),dtype=np.uint8)
        input_image[0:row,0:col]=image

        INPUT_WH_YOLO=640
        blob=cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        preds=self.yolo.forward()
        
        detections=preds[0]
        boxes=[]
        confidences=[]
        classes=[]

        image_w,image_h=input_image.shape[:2]
        x_factor=image_w/INPUT_WH_YOLO
        y_factor=image_h/INPUT_WH_YOLO

        for i in range(len(detections)):
            row=detections[i]
            confidence=row[4]
            if confidence>0.3:
                class_score=row[5:].max()
                class_id=row[5:].argmax()

                if class_score>0.15:
                    cx,cy,w,h=row[0:4]
                    left=int((cx-0.5*w)*x_factor)
                    top=int((cy-0.5*h)*y_factor)
                    width=int(w*x_factor)
                    height= int(h*y_factor)
                    box=np.array([left,top,width,height])
            
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np=np.array(boxes).tolist()
        confidences_np=np.array(confidences).tolist()
        # indox=np.concatenate(boxes_np,confidences_np)
        indax=cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
        for ind in indax:
            x,y,w,h=boxes_np[ind]
            bb_conf=int(confidences_np[ind]*100)
            classes_id=classes[ind]
            class_name=self.labels[classes_id]

            text=f'{class_name}:{bb_conf}%'
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(image,(x,y-30),(x+w,y),(255,255,255),-1)

            cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
            
        cv2.imshow('Original',img)
        # cv2.imshow('Plant_Prediction',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image