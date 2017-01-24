# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Runs the entire backend for the leaf classification, image
# processing/segmentation, and feature extraction

import datamaker as dm
from extraction import ExtractFeatures
from classification import Classifier

import matplotlib.image as mpimg


# --- Training ----------------------------------------------------------------

def train():
    """Run the program in order to train the model"""
    # Populate a data file with features
    dm.populate_training_data()
    dm.populate_test_data()

    # Initiate and train the model
    clf = Classifier()
    clf.train()
    clf.test()


# --- Operating ---------------------------------------------------------------

def run():
    """
    The main function for the species classification portion of the program.

    Will take an image as input and run it through:
    1. Image segmentation
    2. Feature extraction
    3. Species classification
    """

    # 1. Perform image segementation and extract a binary image
    img = mpimg.imread('input/images/4.jpg')

    # 2. Extract features from the image
    features_object = ExtractFeatures(img)
    features = features_object.features

    # 3. Create a prediction from the features
    clf = Classifier()
    clf.load_model()
    predictions = clf.predict(features)

    # Return the predictions
    return predictions

# pred = run()
# print(pred)
train()
