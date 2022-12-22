def UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k):
	"""
	parameters:
	imgPath: the path of the image folder. Please use relative path
	imgFilename: the name of the image file
	savedImgPath: the path of the folder you will save the image
	savedImgFilename: the name of the output image
	k: the number of clusters of the k-means function
	function: using k-means to segment the image and save the result to an image with a bounding box
	"""
	if __name__ == "__main__":
		imgPath=""
		imgFilename="face_d2.jpg"
		savedImgPath=r'face_d2.jpg'
		savedImgFilename="face_d2_face.jpg"
		k=3
	import cv2
	import numpy as np



UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k)



