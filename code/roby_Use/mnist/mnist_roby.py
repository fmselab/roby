"""
In this **MNIST classification** example we test the robustness a
CNN used to identify manual written numbers. This is a widely used
case study, for which the images are freely available in the Keras
Python Library

In this example:

- The _images_ are loaded as a list of np.arrays from the function
  `mnist.load_data()`
- The _classes_ are loaded manually inside a list.
- The _labels_ are loaded as a list of strings, from the function
  `mnist.load_data()`
"""
from models.mobilenet import MobileNet   # type: ignore
from keras.datasets import mnist   # type: ignore
from roby.EnvironmentRTest import EnvironmentRTest
import cv2   # type: ignore
import numpy as np   # type: ignore
from roby.Alterations import Compression, GaussianNoise,\
    VerticalTranslation, HorizontalTranslation, Blur, Brightness, Zoom,\
    Alteration
from roby.RobustnessCNN import robustness_test,\
    display_robustness_results, classification


def pre_processing(image):
    """
    Pre-process the image for classification, in the same way of the pictures
    used to train the CNN
    """
    image = cv2.resize(image, (28, 28))
    if (image.ndim == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=0)
    if not(len(np.shape(image)) == 3 and np.shape(image)[2] == 1):
        image = np.expand_dims(image, axis=-1)

    return image


def load_dataset():
    """
    Loads the train and test dataset. Since we start from an already trained
    CNN, we are interested only in the train one.
    """
    # load dataset
    (testX, testY) = mnist.load_data()[1]
    # reshape dataset to have a single channel
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # convert from integers to floats
    test_norm = testX.astype('float32')
    # normalize to range 0-1
    test_norm = test_norm / 255.0
    # return normalized images
    return test_norm, testY


if __name__ == '__main__':
    # load the model
    model = MobileNet()
    print('Loading pre-trained weights for mobile net')
    model.load_weights('models\\MobileNet.h5')
    print('model loaded')
    # set the accuracy threshold
    accuracy_threshold = 0.8
    # get the images in the test data-set
    # -> The method mnist.load_data() returns a couple of values:
    # X_test is the set of images, while y_test is the set of labels
    (X_test, y_test) = load_dataset()

    # set the classes
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # load the environment
    environment = EnvironmentRTest(model, X_test[:100], classes,
                                   preprocess_f=pre_processing,
                                   label_list=y_test[:100])

    # get the standard behavior of the net
    accuracy = classification(environment)

    # create the alteration_type as a GaussianNoise with variance 0.01
    alteration_type: Alteration = GaussianNoise(0, 1, 0.01)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Compression
    alteration_type = Compression(0, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Vertical Translation
    alteration_type = VerticalTranslation(-1, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Horizontal Translation
    alteration_type = HorizontalTranslation(-1, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Blur Variation with radius = 2,
    # with black and white images (L)
    alteration_type = Blur(0, 1, 2, 'L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Brightness Variation,
    # with black and white images (L)
    alteration_type = Brightness(0, 1, 'L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Zoom
    alteration_type = Zoom(0, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)
