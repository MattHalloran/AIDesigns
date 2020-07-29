#Original code from: https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066

import cv2
from cv2 import dnn_superres
import os

dirname = os.path.dirname(__file__)
scale = 8
filename = 'elon-head.png'

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread(f'{dirname}/Input/{filename}')

# Read the desired model
path = f'{dirname}/Model/LapSRN_x8.pb'
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("lapsrn", scale)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite(f'{dirname}/Output/{filename}', result)

