{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "5111b662-8170-4b6d-b834-c642db687c48",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "import yaml\n",
    "from yaml.loader import SafeLoader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "3b9377dc-99ec-4347-ba1e-c11221f6478e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Bacterial_Spot']\n"
     ]
    }
   ],
   "source": [
    "with open('data.yaml',mode='r') as f:\n",
    "    data_yaml=yaml.load(f,Loader=SafeLoader)\n",
    "\n",
    "labels=data_yaml['names']\n",
    "print(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "6884cc89-97cc-45e1-82de-93a67646085a",
   "metadata": {},
   "outputs": [],
   "source": [
    "yolo=cv2.dnn.readNetFromONNX('C:/Users/DELL/OneDrive/Desktop/Plant Disease Prediction/data_preparation/predictions/Model2/weights/best.onnx')\n",
    "yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)\n",
    "yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "f2507f4b-86ce-4cb6-912f-647a1665618d",
   "metadata": {},
   "outputs": [],
   "source": [
    "img=cv2.imread('C:/Users/DELL/OneDrive/Desktop/Plant Disease Prediction/data_preparation/predictions/img3.jpg')\n",
    "image=img.copy()\n",
    "# cv2.imshow('image',image)\n",
    "# cv2.waitKey(0)\n",
    "# cv2.destroyAllWindows()\n",
    "row,col,d=image.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "de1a13d7-a542-4aff-a62f-333756ceb749",
   "metadata": {},
   "outputs": [],
   "source": [
    "max_rc=max(row,col)\n",
    "input_image=np.zeros((max_rc,max_rc,3),dtype=np.uint8)\n",
    "input_image[0:row,0:col]=image\n",
    "\n",
    "INPUT_WH_YOLO=640\n",
    "blob=cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)\n",
    "yolo.setInput(blob)\n",
    "preds=yolo.forward()\n",
    "\n",
    "# cv2.imshow('input_image',input_image)\n",
    "# cv2.waitKey(0)\n",
    "# cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "e3f4ea65-6370-4276-9bd7-8ddfa0e6be25",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1, 25200, 6)\n"
     ]
    }
   ],
   "source": [
    "print(preds.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "29287b89-8813-4a2d-b5fc-2ea0eac0c608",
   "metadata": {},
   "outputs": [],
   "source": [
    "detections=preds[0]\n",
    "boxes=[]\n",
    "confidences=[]\n",
    "classes=[]\n",
    "\n",
    "image_w,image_h=input_image.shape[:2]\n",
    "x_factor=image_w/INPUT_WH_YOLO\n",
    "y_factor=image_h/INPUT_WH_YOLO\n",
    "\n",
    "for i in range(len(detections)):\n",
    "    row=detections[i]\n",
    "    confidence=row[4]\n",
    "    if confidence>0.3:\n",
    "        class_score=row[5:].max()\n",
    "        class_id=row[5:].argmax()\n",
    "\n",
    "        if class_score>0.15:\n",
    "            cx,cy,w,h=row[0:4]\n",
    "            left=int((cx-0.5*w)*x_factor)\n",
    "            top=int((cy-0.5*h)*y_factor)\n",
    "            width=int(w*x_factor)\n",
    "            height= int(h*y_factor)\n",
    "            box=np.array([left,top,width,height])\n",
    "    \n",
    "            confidences.append(confidence)\n",
    "            boxes.append(box)\n",
    "            classes.append(class_id)\n",
    "\n",
    "boxes_np=np.array(boxes).tolist()\n",
    "confidences_np=np.array(confidences).tolist()\n",
    "\n",
    "index=cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "ddb65b9e-674c-4c20-816f-edd36673e04d",
   "metadata": {},
   "outputs": [],
   "source": [
    "for ind in index:\n",
    "    x,y,w,h=boxes_np[ind]\n",
    "    bb_conf=int(confidences_np[ind]*100)\n",
    "    classes_id=classes[ind]\n",
    "    class_name=labels[classes_id]\n",
    "\n",
    "    text=f'{class_name}:{bb_conf}%'\n",
    "    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)\n",
    "    cv2.rectangle(image,(x,y-30),(x+w,y),(255,255,255),-1)\n",
    "\n",
    "    cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "37e2cc2d-3995-4ba2-82cb-f81761d0686a",
   "metadata": {},
   "outputs": [],
   "source": [
    "cv2.imshow('Original',img)\n",
    "cv2.imshow('Plant_Prediction',image)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efc68d3b-b426-4b3f-8419-3b78d3cec95c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "898b82c1-0758-4df1-96e4-d8a306f1cd7a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42f8ca65-b729-492a-8d14-119dd424b46c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e2d629f-1540-462e-9713-77200a05353a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4369f460-93b3-4a83-8c0a-d96b7b7a76f5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efb4d8b0-0f1a-4e8c-912b-aa40d4a38f9f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8eec7c1f-5fc6-450e-854b-13c59faa0121",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "885d3e67-5e42-4507-aa90-6f7a8a25aaf1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76e1c163-6fec-4505-b958-17c30eef3af7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "PlantDP",
   "language": "python",
   "name": "plantdp"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
