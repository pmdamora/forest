# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Extracts numeric data from input images of leaves
# Expects a processed image where the area of the leaf is white, and the rest
# of the image is white. Any further necessary image preprocessing is performed
# here.
#
# Shape Features: Eccentricity, Aspect Ratio, Area ratio of perimeter,
#   Solidity, Convexity, Elongation, Isoperimetric Factor, Leaf Area,
#   Leaf Perimeter
#
#
# References:
# http://iosrjournals.org/iosr-jece/papers/Vol.%2010%20Issue%205/Version-1/Q01051134140.pdf
# http://alumni.cs.ucr.edu/~lexiangy/Shapelet/kdd2009shapelet.pdf
# https://www.researchgate.net/publication/266632357_Plant_Leaf_Classification_using_Probabilistic_Integration_of_Shape_Texture_and_Margin_Features?enrichId=rgreq-2d37f571dd9abb07ac32daf5e53a9c64-XXX&enrichSource=Y292ZXJQYWdlOzI2NjYzMjM1NztBUzoxNjYyNzk4MDk4NzE4NzJAMTQxNjY1NTYwNDk1Nw%3D%3D&el=1_x_2
# https://www.kaggle.com/lorinc/leaf-classification/feature-extraction-from-images/notebook
# http://www.math.uci.edu/icamp/summer/research_11/park/shape_descriptors_survey.pdf

import numpy as np
import scipy as sp
import scipy.ndimage as ndi
import csv
from skimage import measure
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
# from matplotlib.patches import Ellipse  # For plotting ellipse
from scipy.spatial import ConvexHull
import pandas as pd


class ExtractFeatures:
    def __init__(self, img=None, test=0):
        # If an image number is included, we are performing a test
        if test > 0:
            self.title, self.img = list(self.get_images([test]))[0]

        # If an image is included, we can continue on with the feature
        # extraction. No title is needed.
        if img is not None:
            self.img = self.preprocess(img)

        # Either way, let's go ahead and extract some stuff
        self.extract()

    def __str__(self):
        (ecc, aspect, a_ratio, solidity, convexity, elongation, isoperimetric,
            perim, area_s) = self.features
        return "Eccentricity: {:f} \nAspect Ratio: {:f}\
               \nArea Ratio of Perimeter: {:f} \nSolidity: {:f}\
               \nConvexity: {:f} \nElongation: {:f}\
               \nIsoperimetric Factor: {:f} \nPerimeter: {:f} \nArea: {:f}"\
               .format(ecc, aspect, a_ratio, solidity, convexity, elongation,
                       isoperimetric, perim, area_s)

    def extract(self):
        """
        The main method for the ExtractFeatures class.

        extract() expects that the variable self.img contains an image of
        a leaf
        """
        # Smooth the edges of the image
        blur = self.extract_shape(self.img)

        # Information needed to extract features, but not features themselves
        contour = self.get_contour(self.img)
        shape = self.get_contour(blur)  # contour of the smoothed image
        edge_x, edge_y = self.coords_to_cols(contour)

        # Defines a best fit ellipse for a set of points
        # In this case, the contour of the image
        (center, phi, axes) = self.find_ellipse(edge_x, edge_y)
        hull = ConvexHull(contour)  # non-smoothed
        area = self.polygon_area(contour)  # non-smoothed
        perim_s = self.find_perimeter(shape)  # smothed

        # Calling the feature extraction function to find individual features
        perim = self.find_perimeter(contour)  # non-smoothed
        area_s = self.polygon_area(shape)  # smoothed
        ecc = self.find_eccentricity(axes)  # of best-fitting ellipse
        aspect = self.find_aspect_ratio(axes)  # of best-fitting ellipse
        a_ratio = area_s/perim_s
        solidity = self.find_solidity(area_s, shape)  # both smoothed
        convexity = self.find_perimeter(contour[hull.vertices])/perim
        elongation = self.find_elongation(axes)
        isoperimetric = self.find_isoperimetric(area, perim)  # both unsmoothed
        # indentation = self.find_max_indentation_depth(hull, contour)

        # Save the features as an instance variable
        self.features = (ecc, aspect, a_ratio, solidity, convexity, elongation,
                         isoperimetric, perim, area_s)

        # Plot the best fitting ellipse
        # ell = Ellipse(xy=center, width=2*axes[0], height=2*axes[1],angle=phi)
        # plt.gca().add_patch(ell)

        # Plot the contour around the image
        # plt.plot(edge_x, edge_y,  'r', linewidth=1.5)

    # --- I/O -----------------------------------------------------------------

    def insert_in_file(self, species=None):
        """
        Inserts the features extracted by an image into a file ready to be read
        by a NN.
        """
        (ecc, aspect, a_ratio, solidity, convexity, elongation, isoperimetric,
         perim, area_s) = self.features

        if species:
            filename = './input/train.csv'
        else:
            filename = './input/test.csv'

        with open(filename, 'a') as data_file:
            file_writer = csv.writer(data_file)
            if species:
                row = [self.title, species, ecc, aspect, a_ratio, solidity,
                       convexity, elongation, isoperimetric, perim, area_s]
            else:
                row = [self.title, ecc, aspect, a_ratio, solidity, convexity,
                       elongation, isoperimetric, perim, area_s]
            file_writer.writerow(row)
        data_file.close()

    def load_image(self, img_num, img_dir='./input/images/', img_type='.jpg'):
        """Reads an image into a numpy array"""
        return mpimg.imread(img_dir + str(img_num) + img_type)

    def get_images(self, num, avail_images=1584):
        """
        Accepts either an integer value or a list of integer values.

        If the value is an integer, we assume it is a count of images
        to sample. We randomly sample the count from the total available
        images, and yield the numpy arrays for each image.

        If num is a list of integers, we yield the numpy array for each
        image number specified.

        :param num: the image title or the number of images to retrieve
        :param avail_images: the total number of available images
        :rval: the number of each image being loaded
        :rval: a numpy array of each image
        """
        if type(num) == int:
            images = range(1, avail_images)
            # Chooses a random sample from the total available number of images
            num = np.random.choice(imgs, size=num, replace=False)

        for img_num in num:
            yield img_num, self.preprocess(self.load_image(img_num))

    # --- Image Displays ------------------------------------------------------

    def display_image(self):
        """Displays the leaf image"""
        plt.imshow(self.img, cmap='gist_ncar_r')
        plt.show()

    # --- Preprocessing -------------------------------------------------------

    def preprocess(self, img):
        """Calls the preprocessing functions & returns a preprocessed image"""
        img = self.portrait(img)
        img = self.resample(img)
        img = self.fill(img)
        img = self.threshold(img)
        return img

    def threshold(self, img, threshold=250):
        return ((img > threshold) * 255).astype(img.dtype)

    def portrait(self, img):
        """Convert all landscape images to portrait."""
        y, x = np.shape(img)
        return np.flipud(img.transpose()) if x > y else img

    def resample(self, img, size=500):
        """Resamples img to size without distortion"""
        ratio = size / max(np.shape(img))
        return sp.misc.imresize(img, ratio, mode='L', interp='nearest')

    def fill(self, img, size=500, tolerance=0.95):
        """
        Extends the image if it is signifficantly smaller than size.

        :param size: expected size of the image in both x and y.
        :param tolerance: tolerance for deviation from size.
        :rval: an array padded with images so that its size is equal to size
        """
        y, x = np.shape(img)

        if x <= size * tolerance:
            pad = np.zeros((y, int((size - x) / 2)), dtype=int)
            img = np.concatenate((pad, img, pad), axis=1)

        if y <= size * tolerance:
            pad = np.zeros((int((size - y) / 2), x), dtype=int)
            img = np.concatenate((pad, img, pad), axis=0)

        return img

    # --- Postprocessing ------------------------------------------------------

    def get_contour(self, img):
        """Find the longest contour in the image."""
        return max(measure.find_contours(img, .8), key=len)

    def get_center(self, img):
        """Finds the center of mass of the leaf"""
        return ndi.measurements.center_of_mass(img)

    def get_centeroid(self, arr):
        """Get the centroid of a set of points"""
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length

    def coords_to_cols(self, points):
        """Converts x,y pairs to feature columns"""
        x = points[::, 1]
        y = points[::, 0]
        return x, y

    def extract_shape(self, img):
        """
        Expects prepared image, returns leaf shape in img format.
        The strength of smoothing had to be dynamically set
        in order to get consistent results for different sizes.
        """
        size = int(np.count_nonzero(img)/1000)
        brush = int(5 * size/size**.75)
        blur = ndi.gaussian_filter(img, sigma=brush, mode='nearest') > 200
        return ((blur) * 255).astype(img.dtype)

    # --- Feature Engineering -------------------------------------------------

    def find_ellipse(self, x, y):
        """
        Wrapper function for all the function calls to the find ellipse
        methods.

        Finds the best fit ellipse to a series of data points describing the
        contour of a leaf.

        References:
        http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html#the-approach
        http://mathworld.wolfram.com/Ellipse.html
        https://stackoverflow.com/questions/13635528/fit-a-ellipse-in-python-given\
        -a-set-of-points-xi-xi-yi

        :param x: a numpy array of x coordinates
        :param y: a numpy array of y coordiantes
        :rval center: the x and y coordinates of the center of the ellipse
        :rval phi: the angle of rotation of the ellipse
        :rval axes: the axes of the ellipse
        """
        xmean = x.mean()
        ymean = y.mean()
        x -= xmean
        y -= ymean
        a = self.fit_ellipse(x, y)
        center = self.ellipse_center(a)
        center[0] += xmean
        center[1] += ymean
        phi = self.ellipse_angle_of_rotation(a)
        axes = self.ellipse_axis_length(a)
        x += xmean
        y += ymean
        return center, phi, axes

    def fit_ellipse(self, x, y):
        """
        Given a series of data points, finds a, which is then used to
        compute the describing characteristics of an ellipse.

        :param x: a numpy array of demeaned x coordinates
        :param y: a numpy array of demeaned y coordinates
        """
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    def ellipse_center(self, a):
        """Given a, find the center of the ellipse"""
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        x0 = (c*d-b*f)/num
        y0 = (a*f-b*d)/num
        return np.array([x0, y0])

    def ellipse_angle_of_rotation(self, a):
        """Given a, find the angle of rotation of the ellipse"""
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        if b == 0:
            if a > c:
                return 0
            else:
                return np.pi/2
        else:
            if a > c:
                return np.arctan(2*b/(a-c))/2
            else:
                return np.pi/2 + np.arctan(2*b/(a-c))/2

    def ellipse_axis_length(self, a):
        """Given a, find the axis length of the ellipse"""
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = np.absolute(2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g))
        down1 = np.absolute((b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c))) -
                            (c+a)))
        down2 = np.absolute((b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c))) -
                            (c+a)))

        res1 = np.sqrt(up/down1)
        res2 = np.sqrt(up/down2)

        return np.array([res1, res2])

    def find_eccentricity(self, axes):
        """Find the eccentricity of an ellipse, given it's axes"""
        a = max(axes)
        b = min(axes)
        e = np.sqrt(1 - (b**2/a**2))

        return e

    def find_aspect_ratio(self, axes):
        """Find the aspect ratio of an ellipse, given it's axes"""
        a = max(axes)
        b = min(axes)
        ratio = b/a

        return ratio

    def find_perimeter(self, points):
        """This function only exists for conformity's sake"""
        temp = None
        length = 0
        for i, point in enumerate(points):
            if i == 0:
                temp = point
            else:
                length += np.linalg.norm(point-temp)
                temp = point
        return length

    def find_solidity(self, area, shape):
        """
        Calculates the solidity of a shape.
        Expects the area of the shape, and the shape itself as input.

        :param area: the area of the shape
        :param shape: the shape, which is a smoothed leaf image
        """
        hull = ConvexHull(shape)
        hull_area = self.polygon_area(shape[hull.vertices])
        solidity = area/hull_area

        return solidity

    def polygon_area(self, points):
        """Calculates the area of any polygon"""
        lines = np.hstack([points, np.roll(points, -1, axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
        return area

    def find_elongation(self, axes):
        """
        Find the elongation of an ellipse
        References:
        http://www.math.uci.edu/icamp/summer/research_11/park/shape_descriptors_survey_part2.pdf
        (slide 17)
        """
        a = max(axes)
        b = min(axes)
        elo = 1 - b/a
        return elo

    def find_isoperimetric(self, area, perim):
        """Finds the isoperimetric factor"""
        isoperimetric = (4 * np.pi * area)/(perim)**2
        return isoperimetric

    def find_max_indentation_depth(self, hull, contour):
        hull_contour = contour[hull.vertices]
        hull_length = self.find_perimeter(hull_contour)
        c_h = self.get_centeroid(hull_contour)
        X, temp_y = self.coords_to_cols(hull_contour)
        temp_x, Y = self.coords_to_cols(contour)

        x_distances = np.zeros(X.shape)
        y_distances = np.zeros(Y.shape)
        # Compute d(X, CH(I)) for each X
        for i, x in enumerate(X):
            x_distances[i] = np.linalg.norm(x-c_h)
        # Compute d(Y, CH(I)) for each Y
        for i, y in enumerate(Y):
            y_distances[i] = np.linalg.norm(y-c_h)

        # Find the maximal distances
        d_x = max(x_distances)
        d_y = max(y_distances)

        # Compute the maximum indentation depth
        indentation = (d_x - d_y)/hull_length

        return indentation

ex = ExtractFeatures(test=194)
print(ex)
