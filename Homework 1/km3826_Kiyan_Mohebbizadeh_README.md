Name: Kiyan Mohebbizadeh 

UNI: km3826

ASSIGNMENT 1 (K-Means: Face Detection)

In this assignment we were required to identify the faces in the
image using kmeans to group similar color values. 

the process looks something like this:

- load the image
- run any filters on the color values such as blurring, median filters
pixelation, etc.
- process the image using k-means to group similar color values 
together
- segment the image based on the k-means processed the image to extract 
only the face 
- create a boundary box with the segmented face
- overlay the box onto the original image

MY PROCESS:

I chose to use HSV color values for this assignment primarily because 
they are the easiest to manipulate using PYTHON's open-cv package. The 
Hue, Saturation and Vibrance factors allowed me limit the range of values
in the image allowing k-means to focus on the Hue and vibrance of the image
which alone can make up most of the image (much like removing L from
LAB values). Once the image was preprocessed I then ran the k-means 
algorithm on the image. I chose higher "k" values to allow for more of the 
definition and color variation to show through for late in the process.
Once kmeans had been run, I then converted the image to grayscale. 
Having the image in gray scale allows me to pick the highest valued 
pixels from a single number and set it as the threshold. And Because of 
the preprocessing and the nature of the image, the faces are among the highest values in grayscale.
Once the image was in grayscale, I ran kernel filters through the image
once again to remove the noise in the image. For example, the hair in 
the bottom left side of the single face image shows as a high value in 
the grayscale, so if I were to convert the image to binary before
the filter, the hair would show up in the binary image. By running the 
filter over the image I was able to level out some of the small areas that 
had higher values, making the faces the highest values the sole highest
value in the image. From here I set the boundary box according to the
binary segmented image of the face and overlaid the this onto the original 
image

MULTIPLE FACE EXTRA CREDIT:

For this part of the assignment I had to adjust a couple factors from 
the single-face function for it to work on the multiple face image. Here
is what I changed:

- There is part of this function that removes some of the values from 
the image. Because of the nature of the image, many of the color values are very 
similar, so I was able to remove some of the values around the left side
of subjects apparel to simplify the k-means search
- There is alot more texture and variation in this image, to combat this
I did have to use stronger and more frequent blur and pixelation filters
- Because the color values are so similar, the sensitivity to these
values must be accounted for. That is why in this image, the actual
bounding boxes of the faces do not encompass the whole face. As the color
values change according to the lighting and from person to person, the
the peripheral face values match those around other parts of the image.
with this relatively high variation, we are unable to grasp the whole face 
value from color alone.
- The bounding box part of this function also had to account for multiple 
boxes for the multiple faces.

LIMITATIONS:

Although these algorithms are slightly generalizable to similar
images, there are several factors that limit the scalability. 

- if the faces are of different complexion then the method of picking
the lightest values from the grayscale my not be possible
- if the background of the image or the lighting conditions of the image
change then there may be more difficulty in detecting the faces in those image
(we saw some of these difficulties come to fruition in the extra credit 
problem)
- one of the methods I used was to filter out the noise was to remove smaller 
areas that were considered "noise" but in certain images removing the small
areas my not be possible depending on surrounding colors and the makeup of the image.
- One factor that helped with the distinction of the faces in the secind image was
the fact that the subjects had hair and hats. If there were to be a bald individual 
with the background from the extra credit image, it may be more difficult to single out
the faces

IMPROVEMENT:

- to improve this script we could take into account the overall color scheme of an image 
to set the binary thresholds accordingly
- another factor could be to take shape into account. By having a general "shape"
of a face we could segment the faces more effectively
- I think that using functions like this to create a training set for a Machine Learning 
Algorithm could be the best solution. Allowing the computer to learn what face color
face shape, etc. we could most accurately detect faces.


