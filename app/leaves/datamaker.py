# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Creates data files for model training and testing

from extraction import ExtractFeatures
import pandas as pd
import csv


def populate_training_data():
    """Populates a file with data ready for training (based on Kaggle split)"""
    data = pd.read_csv('input/train_kaggle.csv')
    id = data.pop('id')
    y = data.pop('species')

    create_data_file()

    for index, num in enumerate(id):
        f = ExtractFeatures(test=num)
        f.insert_in_file(y[index])


def populate_test_data():
    """Populates a file with data ready for testing (based on Kaggle split)"""
    data = pd.read_csv('input/test_kaggle.csv')
    id = data.pop('id')

    create_data_file(False)

    for index, num in enumerate(id):
        f = ExtractFeatures(test=num)
        f.insert_in_file()


def create_data_file(include_species=True):
    """Creates a csv file and populates with headers for data"""
    if include_species:
        filename = './input/train.csv'
    else:
        filename = './input/test.csv'

    with open(filename, 'w') as data_file:
        file_writer = csv.writer(data_file)

        if include_species is True:
            file_writer.writerow(['id', 'species', 'eccentricity',
                                  'aspect ratio', 'apratio', 'solidity',
                                  'convexity', 'elongation',
                                  'isoperimetric', 'perim', 'area'])
        else:
            file_writer.writerow(['id', 'eccentricity', 'aspect ratio',
                                  'apratio', 'solidity', 'convexity',
                                  'elongation', 'isoperimetric',
                                  'perim', 'area'])
    data_file.close()
