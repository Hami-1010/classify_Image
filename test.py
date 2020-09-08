from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os



LABELS = set(["football"])

# Load image de xu ly anh
imagePaths = list(paths.list_images("data"))
data = []
labels = []
# Loop thu muc data

for imagePath in imagePaths:

	label = imagePath.split(os.path.sep)[-2]
	
	# Xu ly anh
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# Them vao du lieu data va label
	data.append(image)
	labels.append(label)
print(len(labels))