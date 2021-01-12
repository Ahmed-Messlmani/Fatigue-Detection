from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
import os
from matplotlib import pyplot
import mtcnn
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from threading import Thread
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import h5py

import matplotlib.pyplot as plt
from keras import backend as K
import base64
from flask import Flask,render_template,url_for,request,flash,jsonify,Response
from flask_bootstrap import Bootstrap 
from imutils.video import VideoStream
import h5py
from PIL import Image
from io import BytesIO
import argparse
import time
import imutils
import base64
import re
import json
import cv2
import playsound
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'
Bootstrap(app)


@app.route('/',methods=['GET'])
def home():
	return render_template('Home.html')

@app.route('/index',methods=['POST'])
def index():
		
	#if request.method == 'POST':

		#if(request.form['index']=='take a picture from laptop'):
		#	return render_template('index.html')
		#else :
		#	return render_template('base64.html')
	#else:

		return render_template('index.html')

@app.route('/base',methods=['POST'])
def base():
	return render_template('base64.html')
@app.route('/streaming',methods=['POST'])
def streaming():
	flash('you need to wait about 15 seconds of loading model so the video camera shows on this page ')
	return render_template("streaming.html")

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (150, 150))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
def generate():
# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = "C:/Users/ahmed/OneDrive/Bureau/projet/fatiguedetection/face_detector/deploy.prototxt"
	weightsPath ="C:/Users/ahmed/OneDrive/Bureau/projet/fatiguedetection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	maskNet = load_model("D:/projectIA/face-mask-detector/videostreaming")
	print("[INFO] starting video stream...")
	global vs 
	vs = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	time.sleep(2.0) 
	

# loop over the frames from the video stream
	while True:
		check,frame = vs.read()
		frame = imutils.resize(frame, width=400)
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		
		try:
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(Fatigue, NonFatigue) = pred
				

				# determine the class label and color we'll use to draw
				# the bounding box and text
				label = "Fatigue" if Fatigue > NonFatigue else "Non Fatigue"
				
				if label == "Fatigue":
					color = (0, 0, 255)
					
				else:
					color=(0, 255, 0)
					
				# include the probability in the label
				label = "{}".format(label)

				# display the label and bounding box rectangle on the output
				# frame
				

				cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			ret, jpeg = cv2.imencode('.jpg', frame)
			frame=jpeg.tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		except Exception as e:
			raise e




@app.route("/video_feed" )
def video_feed():
	return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed1" , methods=['POST'])
def video_feed1():
	global vs
	vs.release()
	return render_template('Home.html')
	

@app.route('/predictindex', methods=['POST'])
def predictindex():

	

	
	
	# Receives the input query from form
	if request.method == 'POST':

		element = request.files['mydata']
		name = element.filename
		#img = base64.b64encode(element.read())
		#filename = os.path.join('C:/Users/ahmed/OneDrive/Images/Pellicule/'+element) 
		# load image from file
		img = pyplot.imread(element)
		# create the detector, using default weights
    	
		detector = MTCNN()
		# detect faces in the image
		face = detector.detect_faces(img)
		# display faces on the original image
		x1, y1, width, height = face[0]['box']
		x2, y2 = x1 + width, y1 + height
		pyplot.axis('off')
    
		pyplot.imshow(img[y1:y2,x1:x2])
		pyplot.savefig('C:/Users/ahmed/OneDrive/Bureau/test image/'+ name)
		#path="C:/Users/ahmed/Downloads/saved_model/"+ new_image
		path=os.path.join("C:/Users/ahmed/OneDrive/Bureau/test image", name)
		img = image.load_img(path, target_size=(150, 150))
		img_tensor = image.img_to_array(img)
		img_tensor = np.expand_dims(img_tensor, axis=0) 
		img_tensor /= 255.
		                                     
		#data = [namequery]
		#vect = cv.transform(data).toarray()
		new_model = tf.keras.models.load_model('C:/Users/ahmed/OneDrive/Bureau/projet/fatiguedetection/models/modeltransfertlearning')

		resultat = new_model.predict(img_tensor)
		if(resultat[0]>0.5):
			resultat[0]=1
		else:
			resultat[0]=0
		K.clear_session()
		return render_template('result.html',prediction = resultat[0])
	
@app.route('/predict', methods=['POST'])
def predict():
	
	
	
	# Receives the input query from form
	
	if request.method == 'POST':
		data = request.form
		image_data=data['mydata']
		img = Image.open(BytesIO(base64.b64decode(image_data)))
		rgb_im = img.convert('RGB')
		rgb_im.save("C:/Users/ahmed/OneDrive/Images/Pellicule/ahmed.jpg")
		name="ahmed.jpg"
		#image = base64.b64encode(element.read())
		#filename = os.path.join('C:/Users/ahmed/OneDrive/Images/Pellicule/'+element) 
		# load image from file
		im1 = pyplot.imread("C:/Users/ahmed/OneDrive/Images/Pellicule/ahmed.jpg")
		# create the detector, using default weight
	
		detector = MTCNN()
		# detect faces in the image
		face = detector.detect_faces(im1)
		# display faces on the original image
		x1, y1, width, height = face[0]['box']
		x2, y2 = x1 + width, y1 + height
		pyplot.axis('off')
    
		pyplot.imshow(im1[y1:y2,x1:x2])
		pyplot.savefig('C:/Users/ahmed/OneDrive/Bureau/test image/'+ name)
		#path="C:/Users/ahmed/Downloads/saved_model/"+ new_image
		path=os.path.join("C:/Users/ahmed/OneDrive/Bureau/test image", name)
		img = image.load_img(path, target_size=(150, 150))
		
		img_tensor = image.img_to_array(img)
		print(img_tensor)
		img_tensor = np.expand_dims(img_tensor, axis=0) 
		img_tensor /= 255.
		pred=[]                                     
		#data = [namequery]
		#vect = cv.transform(data).toarray()
		new_model = tf.keras.models.load_model("D:/projectIA/face-mask-detector/videostreaming")
		pred=new_model.predict(img_tensor)
		print (pred)
		
		if(pred[0][0]>pred[0][1]):
			resultat=0
		else:
			resultat=1
		#image = base64.b64encode(element.read())
		#filename = os.path.join('C:/Users/ahmed/OneDrive/Images/Pellicule/'+element) 
		# load image from file
		
		K.clear_session()
		return render_template('result.html',prediction = resultat)
	

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
#ap.add_argument("-m", "--model", type=str,
#	default="mask_detector.model",
#	help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
	args = vars(ap.parse_args())
	app.run(debug=True)

