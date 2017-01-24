# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Builds a model that classifies numeric data as leaf species
#
# References:
# http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://www.kaggle.com/najeebkhan/leaf-classification/neural-network-through-keras
# http://iosrjournals.org/iosr-jece/papers/Vol.%2010%20Issue%205/Version-1/Q01051134140.pdf

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
import pickle

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Define constants, mostly file locations
MODEL_LOC = 'model/model.json'
WEIGHTS_LOC = 'model/model.h5'
STANDARD_LOC = 'model/standards.csv'
NUM_FEATURES = 9
TRAIN_LOC = 'input/train.csv'
TEST_LOC = 'input/test.csv'
SCALAR_LOC = 'model/scaler.pkl'


class Classifier:
    def predict(self, features):
        """
        Expects a tuple of features as input.
        Loads the model, and uses the model to create a set of predictions,
        which are then returned.

        :param features: a tuple of numerical features extracted from an image
        :rval: a set of predictions computed by the classifier
        """
        # Convert the tuple to a numpy array, reshape, and scale the features
        features = np.asarray(features)
        features = features.reshape(1, -1)
        features = self.scaler.transform(features)

        # Create a set of prediction probabilities from the features
        predictions = self.model.predict_proba(features)
        return predictions

    def train(self):
        """Trains the classification model"""
        # Load the training data
        data = pd.read_csv(TRAIN_LOC)
        self.parent_data = data.copy()
        ID = data.pop('id')

        # Create feature and label training datasets
        # The labels are textual (species), so we encode them categorically
        # as 0 to n_classes - 1
        y = data.pop('species')
        y = LabelEncoder().fit(y).transform(y)

        # Standardize the data to give it a mean of 0
        self.scaler = StandardScaler().fit(data)
        X = self.scaler.transform(data)

        # Perform one-hot encoding on the labels
        y_cat = to_categorical(y)

        # Fit the model to the training data
        self.model = self.baseline_model()
        history = self.model.fit(X, y_cat, batch_size=128, nb_epoch=100,
                                 verbose=0)

        # # Plot the error vs. the number of iterations
        # plt.plot(history.history['loss'], 'o-')
        # plt.xlabel('Number of Iterations')
        # plt.ylabel('Categorical Crossentropy')
        # plt.title('Train Error vs Number of Iterations')
        # plt.show()

        # Save the model for later use
        self.save_model()

    def test(self):
        """Loads the model, and tests it"""
        self.load_model()

        # Test the model
        test = pd.read_csv(TEST_LOC)
        index = test.pop('id')

        test = self.scaler.transform(test)

        # Create predictions from the model, and write the predictions to file
        yPred = self.model.predict_proba(test)
        yPred = pd.DataFrame(yPred, index=index,
                             columns=sorted(self.parent_data.species.unique()))
        fp = open('output/submission_kaggle.csv', 'w')
        fp.write(yPred.to_csv())

    def baseline_model(self):
        """Defines a base model"""
        model = Sequential()
        model.add(Dense(1024, input_dim=NUM_FEATURES))  # 1024
        model.add(Dropout(0.2))
        model.add(Activation('sigmoid'))
        model.add(Dense(512))  # 512
        model.add(Dropout(0.3))
        model.add(Activation('sigmoid'))
        model.add(Dense(99))  # 99
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        return model

    def save_model(self):
        """Saves a model to a JSON format for later use"""
        # Serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_LOC, "w") as json_file:
            json_file.write(model_json)

        # Save the scalar
        self.save_object(self.scaler, SCALAR_LOC)

        # Serialize weights to HDF5
        self.model.save_weights(WEIGHTS_LOC)

    def load_model(self):
        try:
            # Open the json file and read it
            json_file = open(MODEL_LOC, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # Load weights into new model
            loaded_model.load_weights(WEIGHTS_LOC)

            # Load the scalar object
            self.load_object(SCALAR_LOC)
        except IOError:
            loaded_model = None
            print("Error loading model: Model could not be found.")
        self.model = loaded_model

    def save_object(self, obj, filename):
        # Save the object to a pickle file
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, -1)

    def load_object(self, filename):
        # Open the pickle file and load the object into self.scaler
        with open(filename, 'rb') as input:
            self.scaler = pickle.load(input)
