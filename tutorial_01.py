# import the necessary packages
from cv2 import cv2

'''***************************************************************'''
'''**************** Loading and displaying an image **************'''
'''***************************************************************'''

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)

image = cv2.imread("good_boys.jpeg") # we assign the result to image . 
                             # Our image  is actually just a NumPy array.
(h, w, d) = image.shape

# Depth is the number of channels — in our case this is three 
# since we’re working with 3 color channels: Blue, Green, and Red.
print("width={}, height={}, depth={}".format(w, h, d))

# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Image", image)
cv2.waitKey(0)

'''***************************************************************'''
'''****************** Accessing individual pixels ****************'''
'''***************************************************************'''

# In OpenCV color images in the RGB (Red, Green, Blue) color space have a 3-tuple
# associated with each pixel: (B, G, R) .
# Notice the ordering is BGR rather than RGB. This is because when OpenCV was first being 
# developed many years ago the standard was BGR ordering. 
# Over the years, the standard has now become RGB but OpenCV still 
# maintains this “legacy” BGR ordering to ensure no existing code breaks.

# Each value in the BGR 3-tuple has a range of [0, 255] . 
# How many color possibilities are there for each pixel in an RGB image in OpenCV? 
# That’s easy: 256 * 256 * 256 = 16777216 .

# access the RGB pixel located at x=50, y=100
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

'''***************************************************************'''
'''****************** Array slicing and cropping *****************'''
'''***************************************************************'''

# Extracting “regions of interest” (ROIs) is an important skill for image processing.
# Say, for example, you’re working on recognizing faces in a movie. 
# First, you’d run a face detection algorithm to find the coordinates of faces in all 
# the frames you’re working with. 
# Then you’d want to extract the face ROIs and either save them or process them.
# For now, let’s just manually extract an ROI. This can be accomplished with array slicing.

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=100 at ending at x=420,y=240
roi = image[100:240, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

'''***************************************************************'''
'''********************* Resizing images *************************'''
'''***************************************************************'''

# Resizing images is important for a number of reasons. 
# First, you might want to resize a large image to fit on your screen. 
# Image processing is also faster on smaller images because there are fewer pixels to process. 
# In the case of deep learning, we often resize images, ignoring aspect ratio, 
# so that the volume fits into a network which requires that an image be square 
# and of a certain dimension.

# resize the image to 200x200px, ignoring aspect ratio
resized = cv2.resize(image, (200, 200)) # resize an image ignoring aspect ratio
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

# Let’s calculate the aspect ratio of the original image and use it to resize it.

# fixed resizing and distort aspect ratio so let's resize the width
# to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)

'''***************************************************************'''
'''********************* Rotating an image ***********************'''
'''***************************************************************'''

# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

'''***************************************************************'''
'''********************* Smoothing an image **********************'''
'''***************************************************************'''

# in many image processing pipelines, we must blur an image to reduce high-frequency noise, 
# making it easier for our algorithms to detect and understand the actual contents of the image 
# rather than just noise that will “confuse” our algorithms. 
# Blurring an image is very easy in OpenCV and there are a number of ways to accomplish it.

# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

'''***************************************************************'''
'''************* Converting an image to grayscale ****************'''
'''***************************************************************'''

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

