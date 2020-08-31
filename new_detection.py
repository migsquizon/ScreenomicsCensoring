
# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import glob
from pytesseract import Output
import re
import enchant
from imutils.object_detection import non_max_suppression
import pytesseract
import string
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import spacy



nlp = spacy.load("en_core_web_sm")
english_dictionary = enchant.Dict("en_US")


def hasNumbers(inputString):
	return bool(re.search(r'\d', inputString))






def anonymize_face_pixelate(image, blocks=5):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image




def drawBoxes(im, boxes):
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	for i in range(x1.shape[0]):
	#     cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
		# print(x1,x2,y1,y2)
		face = im[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
		face = anonymize_face_pixelate(face)
		im[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])] = face

	return im


mtcnn = MTCNN(keep_all=True, device='cpu')



img_dir = 'images' # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
images = glob.glob(data_path)


minsize = 20



threshold = [0.6, 0.7, 0.7]
factor = 0.709



for img in range(len(images)):
	image = cv2.imread(images[img])


	img_matlab = image.copy()
	tmp = img_matlab[:,:,2].copy()
	img_matlab[:,:,2] = img_matlab[:,:,0]
	img_matlab[:,:,0] = tmp

	# boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
	# Detect face
	boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
	# print(boxes)
	if boxes is not None:
		# print(boxes)
		image = drawBoxes(image, boxes)


	image = cv2.resize(image, None, fx=1.2, fy=1.2,interpolation=cv2.INTER_CUBIC )
	kernel = np.ones((1, 1), np.uint8)
	image = cv2.dilate(image, kernel, iterations=1)
	image = cv2.erode(image, kernel, iterations=1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	cv2.threshold(cv2.bilateralFilter(gray, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	cv2.threshold(cv2.medianBlur(gray, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

	cv2.adaptiveThreshold(cv2.bilateralFilter(gray, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

	cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


	results = pytesseract.image_to_data(gray, output_type=Output.DICT)
	for i in range(0, len(results["text"])):
		# extract the bounding box coordinates of the text region from
		# the current result
		x = results["left"][i]
		y = results["top"][i]
		w = results["width"][i]
		h = results["height"][i]

		# extract the OCR text itself along with the confidence of the
		# text localization
		text = results["text"][i]
		conf = int(results["conf"][i])
		# filter out weak confidence text localizations
		if conf > 80:
			# print(text)
			text = re.sub(r'[^\w\s]','',text)
			if text != "":
				# print(text)
				doc = nlp(text)
				# print(text,doc.ents,len(doc.ents))
				if(len(doc.ents)==0):
					if not english_dictionary.check(text) or hasNumbers(text):
						# print("Confidence: {}".format(conf))
						# print("Text: {}".format(text))
						# print("")
						tex = image[y:y+h, x:x+w]
						tex = anonymize_face_pixelate(tex)
						image[y:y+h, x:x+w] = tex
						cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
							1, (0, 0, 255), 2)
					
				else:
					for ent in doc.ents:
						print(ent.text, ent.label_)
						if(ent.label_=='PERSON') or hasNumbers(text) or not english_dictionary.check(text):
							tex = image[y:y+h, x:x+w]
							tex = anonymize_face_pixelate(tex)
							image[y:y+h, x:x+w] = tex
							# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
							cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
								1, (0, 0, 255), 2)




	filename = 'censored/censored_image{}.jpg'.format(img)
	print(filename)
	if not cv2.imwrite(filename,image):
		raise Exception("Could not write image")
	

	