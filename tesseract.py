# USAGE
# python localize_text_tesseract.py --image apple_support.png
# python localize_text_tesseract.py --image apple_support.png --min-conf 50

# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2
import os
import glob
import re
import enchant

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# ap.add_argument("-c", "--min-conf", type=int, default=0,
# 	help="mininum confidence value to filter weak text detection")
# args = vars(ap.parse_args())


img_dir = 'images' # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
images = glob.glob(data_path)

english_dictionary = enchant.Dict("en_US")


def hasNumbers(inputString):
	return bool(re.search(r'\d', inputString))

for img in range(len(images)):
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
	image = cv2.imread(images[img])
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


			# display the confidence and text to our terminal
			# print("Confidence: {}".format(conf))
			# print("Text: {}".format(text))
			# print("")


			if text != "":
				if not english_dictionary.check(text) or hasNumbers(text):
					print("Confidence: {}".format(conf))
					print("Text: {}".format(text))
					print("")

					cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
					cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
						1.2, (0, 0, 255), 3)

			# strip out non-ASCII text so we can draw the text on the image
			# using OpenCV, then draw a bounding box around the text along
			# with the text itself

				# else:
				# 	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
				# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				# 	cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
				# 		1.2, (0, 0, 255), 3)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)