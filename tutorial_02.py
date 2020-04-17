# import the necessary packages
from cv2 import cv2
import numpy as np

'''***************************************************************'''
'''**************** Loading and displaying an image **************'''
'''***************************************************************'''

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)

image = cv2.imread("Tetris_blocks.jpg") 

# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Image", image)
cv2.waitKey(0)

'''***************************************************************'''
'''************* Converting an image to grayscale ****************'''
'''***************************************************************'''

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

'''***************************************************************'''
'''********************* Edge detection **************************'''
'''***************************************************************'''

# Edge detection is useful for finding boundaries of objects in an image — 
# it is effective for segmentation purposes.

# applying edge detection we can find the outlines of objects in images
edged = cv2.Canny(gray, 30, 150)
# We provide three parameters to the cv2.Canny  function:
# img : The gray  image.
# minVal : A minimum threshold, in our case 30 .
# maxVal : The maximum threshold which is 150  in our example.
# aperture_size : The Sobel kernel size. By default this value is 3  and hence is

cv2.imshow("Edged", edged)
cv2.waitKey(0)

'''***************************************************************'''
'''********************* Thresholding ****************************'''
'''***************************************************************'''

# Image thresholding is an important intermediary step for image processing pipelines. 
# Thresholding can help us to remove lighter or darker regions and contours of images.

thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

'''***************************************************************'''
'''************* Detecting and drawing contours ******************'''
'''***************************************************************'''

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()

# loop over the contours
for c in contours:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv2.imshow("Contours", output)
	cv2.waitKey(0)

'''***************************************************************'''
'''************ Draw the total number of objects *****************'''
'''***************************************************************'''

# draw the total number of contours found in purple
text = "I found {} objects!".format(len(contours))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)