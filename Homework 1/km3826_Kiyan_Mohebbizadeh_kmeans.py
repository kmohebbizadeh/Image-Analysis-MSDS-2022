import cv2
import numpy as np
import matplotlib.pyplot as plt

single_face_image = cv2.imread("face_d2.jpg")
multiple_face_image = cv2.imread("faces.jpg")

def single_face_detection(image):
    # save the image as a variable to overlay later, and a variable to edit
    og_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # equalize saturation values to increase the color contrast of the image for K-means
    H,S,V=cv2.split(image)
    S = cv2.equalizeHist(S)
    image = cv2.merge([H,S,V])
    # set the pixel color values in a structure for K-means
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # run K-means on the pixel values.
    # standard criteria for k-means
    # 100 iterations before algorithm stops
    # standard required accuracy of .2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # set k to 7 for seven clusters in the image
    k = 7
    # set the attempts to 10 which is the number of times it runs to find optimality
    # set random centers as we are using classic random center k-means
    # other parameters are standard
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # reformulate the image after running k-means segments it
    centers = np.uint8(centers)
    # create a workable k-means segmented image
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    plt.imshow(segmented_image)
    plt.title("K-means Single Face")
    plt.show()
    # convert the segmented image to grayscale for filtering
    gray = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)
    # once converted to grayscale run a  kernel filter to remove noise in the image
    # we set a medium-sized kernel here to remove the neck and background factors for segmentation
    # morph rect creates more of a pixelation effect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
    # morph open is used to remove small values from background which is what we are looking for
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    plt.imshow(gray)
    plt.title("Gray-Scale Single Face (with pixelating filters)")
    plt.show()
    # after the kernel filter has been run, the binary image will only have the face as a segment
    # set the threshold of 190
    _, binary = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
    plt.imshow(binary)
    plt.title("Binary Single Face")
    plt.show()
    cv2.imwrite('face_d2_binary.jpg', binary)
    # create a bounding box based on the binary image
    x1,y1,w,h = cv2.boundingRect(binary)
    x2 = x1+w
    y2 = y1+h
    start = (x1, y1)
    end = (x2, y2)
    # styling for the boxes
    color = (255, 0, 0)
    thickness = 2
    # apply the bounding box to the original image
    rectangle_img = cv2.rectangle(og_image, start, end, color, thickness)
    plt.imshow(rectangle_img)
    plt.title("Single Face with Bounding Box")
    plt.show()
    rectangle_img = cv2.cvtColor(rectangle_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('face_d2_final.jpg', rectangle_img)


def multiple_face_detection(image):
    # create variables to edit and to overlay. og_image saves the BGR values for the noise removing stage
    og_image = image
    og_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # equalize saturation values to create color contrast in the image
    H,S,V=cv2.split(image)
    S = cv2.equalizeHist(S)
    image = cv2.merge([H,S,V])
    # filter the image with a blur to remove the texture from the image, small kernel is acceptable for this task
    image = cv2.blur(image,(10,10))
    # set the image to grayscale for editing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # this kernel is big enough to remove the noise without making the faces indistinguishable.
    # we use a large kernel to remove the noise which includes the neck and background areas
    # we use the same parameters as mentioned above
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,30))
    # same standard parameters as above
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # convert the image to binary at a threshold fof 107 to bring out the values of obejcts that aren't faces
    _, binary = cv2.threshold(image,107,255,cv2.THRESH_BINARY)
    # once converted to binary with the filters applied, only noise is filtered out.
    # find the countours of the largest object to remove
    # standard parameters for the contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # find the largest object from the contours
    c = max(contours, key = cv2.contourArea)
    # we use an eliptical kernel to dilate the object so that if there are contact points with a face a gap is created
    # 5,5 is a standard kernel for this procedure
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    black = np.zeros((og_image.shape[0], og_image.shape[1]), np.uint8)
    # here we set the shape over the original image
    black = cv2.drawContours(black, [c], 0, 255, -1)
    # once the shape is  set we expand the area slightly to account for the previous dilation
    dilate = cv2.dilate(black, kernel, iterations = 3)
    # with this mask set the values = 255 so that they are removed from consideration for K-means
    og_image[dilate == 255] = (255,255,255)
    plt.imshow(og_image)
    plt.title("Multiple Face (removing similar color values)")
    plt.show()
    # new image has removed some aspects of the image that may confuse kmeans as faces
    # apply the HSV and filters to the new image
    image = og_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # same processing as we did before to make to enhance color contrast
    H,S,V=cv2.split(image)
    S = cv2.equalizeHist(S)
    image = cv2.merge([H,S,V])
    # we are looking for a pretty strong blur due to all the variation in the image that is why we set a large kernel
    image = cv2.blur(image, (30,30))
    # set the pixel color values in a structure for K-means
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # run K-means on the pixel values.
    # standard criteria for k-means mentioned above
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # we have a high k value to maintain the details of this image since it is complex
    k = 16
    # set the attempts to 10 which is the number of times it runs to find optimality
    # set random centers as we are using classic random center k-means
    # standard parameters as mentioned above
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # create a usable k-means segmented image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    # filter out the noise in the gray-scale image
    gray = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY)
    # we use a large kernel for blur and kernel morph here too to reduce small problematic noise areas
    gray = cv2.blur(gray, (20,20))
    # standard parameters with a medium-sized kernel to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16,16))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    plt.imshow(gray)
    plt.title("Gray-Scale Multiple Faces (highest values around the faces) ")
    plt.show()
    # create a binary image with a threshold of 148 to single out the faces
    _, binary = cv2.threshold(gray,148,255,cv2.THRESH_BINARY)
    plt.imshow(binary)
    plt.title("Binary Multiple Faces")
    plt.show()
    cv2.imwrite('faces_binary.jpg', binary)
    # once converted to binary set the boundary boxes based on face segments
    # since there are multiple faces we find the contours of all the objects in the binary image
    # once we have the contours we find the coordinates of the boundary boxes
    # standard parameters for contour search in image
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    colour = (255, 0, 0)
    thickness = 2
    i = 0
    result = og_image1
    # overlay the boundary boxes on the original image
    for cntr in contours:
        x1,y1,w,h = cv2.boundingRect(cntr)
        x2 = x1+w
        y2 = y1+h
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, thickness)
        i += 1

    plt.imshow(result)
    plt.show()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite('faces_final.jpg', result)


single_face_detection(single_face_image)
multiple_face_detection(multiple_face_image)
