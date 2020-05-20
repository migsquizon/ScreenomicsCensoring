# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import glob




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
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


img_dir = 'images' # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
images = glob.glob(data_path)

for img in range(len(images)):
	image = cv2.imread(images[img])
	cv2.imshow("Output", image)
	cv2.waitKey(0)

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
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = image[startY:endY, startX:endX]
			text = "{:.2f}%".format(confidence * 100)
			face = anonymize_face_pixelate(face)
			cv2.putText(image, text, (startX, startY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			image[startY:endY, startX:endX] = face



			# # draw the bounding box of the face along with the associated
			# # probability
			# text = "{:.2f}%".format(confidence * 100)
			# y = startY - 10 if startY - 10 > 10 else startY + 10
			# roi = image[startY:endY, startX:endX]
			# (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			# # cv2.rectangle(image, (startX, startY), (endX, endY),
			# # 	(0, 0, 255), 2)
			# cv2.rectangle(image, (startX, startY), (endX, endY),(B, G, R), -1)
			# cv2.putText(image, text, (startX, y),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# cv2.imshow("Output", image)
	# cv2.waitKey(0)
	filename = 'censored/censored_image{}.jpg'.format(img)
	print(filename)
	if not cv2.imwrite(filename,image):
		raise Exception("Could not write image")
	 


# # show the output image
# cv2.imwrite('censored')

# cv2.waitKey(0)


