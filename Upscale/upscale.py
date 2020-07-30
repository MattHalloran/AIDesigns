#Original code from: https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066

import cv2
from cv2 import dnn_superres
from PIL import Image
from PIL import ImageFilter
import os

dirname = os.path.dirname(__file__)
scaleFactor = 8
filename = 'elon-head'
ext = 'png'

def scaleUp(source:str, dest:str):
    global scaleFactor, filename, dirname

    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read image
    image = cv2.imread(source)

    # Read the desired model
    path = f'{dirname}/Model/LapSRN_x8.pb'
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("lapsrn", scaleFactor)

    # Upscale the image
    result = sr.upsample(image)

    # Save the image
    cv2.imwrite(dest, result)

def sharpen(source:str, dest:str):
    global dirname, filename

    # Open an already existing image
    imageObject = Image.open(source)

    # Apply sharp filter
    sharpened = imageObject.filter(ImageFilter.SHARPEN)

    # Show the sharpened images
    sharpened.save(dest)

def shrink(source:str, dest:str):
    global dirname, filename

    image = Image.open(source)
    width, height = image.size
    image = image.resize((int(width/8), int(height/8)))
    image.save(dest)

#-------------------------Fails---------------------------
#VERY bad
#orig = f'{dirname}/Input/{filename}.{ext}'
#dest = f'{dirname}/Output/{filename}_origShrunk.{ext}'
#shrink(orig, dest)
#scaleUp(dest, dest)
#scaleUp(dest, dest)

#About the same as original image
#orig = f'{dirname}/Input/{filename}.{ext}'
#dest = f'{dirname}/Output/{filename}_justScale.{ext}'
#scaleUp(orig, dest)

#Worse than sharpen first
#orig = f'{dirname}/Input/{filename}.{ext}'
#dest = f'{dirname}/Output/{filename}_sharpenLast.{ext}'
#scaleUp(orig, dest)
#sharpen(dest, dest)

#Starts giving deep-fried effect after first sharpen
#orig = f'{dirname}/Input/{filename}.{ext}'
#dest = f'{dirname}/Output/{filename}_polySharpen.{ext}'
#sharpen(orig, dest)
#sharpen(dest, dest)
#sharpen(dest, dest)
#sharpen(dest, dest)
#sharpen(dest, dest)
#scaleUp(dest, dest)
#-------------------------Fails---------------------------


#A little better than original image
orig = f'{dirname}/Input/{filename}.{ext}'
dest = f'{dirname}/Output/{filename}_sharpenFirst.{ext}'
sharpen(orig, dest)
scaleUp(dest, dest)