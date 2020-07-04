# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

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



english_dictionary = enchant.Dict("en_US")


def hasNumbers(inputString):
	return bool(re.search(r'\d', inputString))


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)




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

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# # ap.add_argument("-p", "--prototxt", required=True,
# # 	help="path to Caffe 'deploy' prototxt file")
# # ap.add_argument("-m", "--model", required=True,
# # 	help="path to Caffe pre-trained model")
# # ap.add_argument("-c", "--confidence", type=float, default=0.5,
# # 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')


img_dir = 'images' # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
images = glob.glob(data_path)

for img in range(len(images)):
	image = cv2.imread(images[img])

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# image = cv2.imread(args["image"])
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = image[startY:endY, startX:endX]
			text = "{:.2f}%".format(confidence * 100)
			face = anonymize_face_pixelate(face)
			# cv2.putText(image, text, (startX, startY),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			image[startY:endY, startX:endX] = face

	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
	# loop over each of the individual text localizations
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
			text = re.sub(r'[^\w\s]','',text)
			if text != "":
				
				if not english_dictionary.check(text) or hasNumbers(text):
					print("Confidence: {}".format(conf))
					print("Text: {}".format(text))
					print("")
					cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
					cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
						1, (0, 0, 255), 2)



	filename = 'censored/censored_image{}.jpg'.format(img)
	print(filename)
	if not cv2.imwrite(filename,image):
		raise Exception("Could not write image")
	

	# orig = image.copy()
	# (origH, origW) = image.shape[:2]

	# # set the new width and height and then determine the ratio in change
	# # for both the width and height
	# (newW, newH) = (320, 320)
	# rW = origW / float(newW)
	# rH = origH / float(newH)

	# # resize the image and grab the new image dimensions
	# image = cv2.resize(image, (newW, newH))
	# (H, W) = image.shape[:2]

	# # define the two output layer names for the EAST detector model that
	# # we are interested -- the first is the output probabilities and the
	# # second can be used to derive the bounding box coordinates of text
	# layerNames = [
	# 	"feature_fusion/Conv_7/Sigmoid",
	# 	"feature_fusion/concat_3"]

	# # load the pre-trained EAST text detector
	# print("[INFO] loading EAST text detector...")
	# net = cv2.dnn.readNet('frozen_east_text_detection.pb')

	# # construct a blob from the image and then perform a forward pass of
	# # the model to obtain the two output layer sets
	# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	# 	(123.68, 116.78, 103.94), swapRB=True, crop=False)
	# net.setInput(blob)
	# (scores, geometry) = net.forward(layerNames)

	# # decode the predictions, then  apply non-maxima suppression to
	# # suppress weak, overlapping bounding boxes
	# (rects, confidences) = decode_predictions(scores, geometry)
	# boxes = non_max_suppression(np.array(rects), probs=confidences)

	# # initialize the list of results
	# results = []

	# # loop over the bounding boxes
	# for (startX, startY, endX, endY) in boxes:
	# 	# scale the bounding box coordinates based on the respective
	# 	# ratios
	# 	startX = int(startX * rW)
	# 	startY = int(startY * rH)
	# 	endX = int(endX * rW)
	# 	endY = int(endY * rH)

	# 	# in order to obtain a better OCR of the text we can potentially
	# 	# apply a bit of padding surrounding the bounding box -- here we
	# 	# are computing the deltas in both the x and y directions
	# 	dX = int((endX - startX) * 0.0)
	# 	dY = int((endY - startY) * 0.0)

	# 	# apply padding to each side of the bounding box, respectively
	# 	startX = max(0, startX - dX)
	# 	startY = max(0, startY - dY)
	# 	endX = min(origW, endX + (dX * 2))
	# 	endY = min(origH, endY + (dY * 2))

	# 	# extract the actual padded ROI
	# 	roi = orig[startY:endY, startX:endX]

	# 	# in order to apply Tesseract v4 to OCR text we must supply
	# 	# (1) a language, (2) an OEM flag of 4, indicating that the we
	# 	# wish to use the LSTM neural net model for OCR, and finally
	# 	# (3) an OEM value, in this case, 7 which implies that we are
	# 	# treating the ROI as a single line of text
	# 	config = ("-l eng --oem 1 --psm 7")
	# 	text = pytesseract.image_to_string(roi, config=config)

	# 	# add the bounding box coordinates and OCR'd text to the list
	# 	# of results
	# 	results.append(((startX, startY, endX, endY), text))

	# # sort the results bounding box coordinates from top to bottom
	# results = sorted(results, key=lambda r:r[0][1])

	# # loop over the results
	# for ((startX, startY, endX, endY), text) in results:
	# 	# display the text OCR'd by Tesseract
	# 	print("OCR TEXT")
	# 	print("========")
	# 	print("{}\n".format(text))

	# 	# strip out non-ASCII text so we can draw the text on the image
	# 	# using OpenCV, then draw the text and a bounding box surrounding
	# 	# the text region of the input image
	# 	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	# 	output = orig.copy()
	# 	cv2.rectangle(output, (startX, startY), (endX, endY),
	# 		(0, 0, 255), 2)
	# 	cv2.putText(output, text, (startX, startY - 20),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

	# 	# show the output image
	# cv2.imshow("Text Detection", output)
	# cv2.waitKey(0) 


# # show the output image
# cv2.imwrite('censored')

# cv2.waitKey(0)


