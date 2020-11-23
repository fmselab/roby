"""
In this **German Traffic Sign classification** example we test the robustness a
CNN used to identify traffic sign. The case study has been taken from this
[paper](https://github.com/francescosecci/Python_Image_Failures/blob/master/Documenti/Paper.pdf),
while the model of which we test the robustness is taken from this
[repository](https://github.com/ItsCosmas/Traffic-Sign-Classification). The
images used to compose the test dataset are taken from this BitBucket
[repository](https://bitbucket.org/jadslim/german-traffic-signs/src/master/).

In this example:

- The _images_ are loaded as a list of np.arrays using a pickle file
  in the `models` folder
- The _classes_ are loaded automatically in a list using the `range()`
  function.
- The _labels_ are loaded as a list of strings, from the same pickle
  file in which is contained the test dataset.
- Only _alterations_ plausible for the application domain are applied.
"""
import cv2   # type: ignore
import numpy as np   # type: ignore
import pickle   # type: ignore
from keras.models import load_model   # type: ignore
from roby.EnvironmentRTest import EnvironmentRTest
from roby.RobustnessNN import classification, robustness_test,\
    display_robustness_results
from roby.Alterations import GaussianNoise, Compression, VerticalTranslation,\
    HorizontalTranslation, Blur, Brightness, Zoom, Alteration
from cameraFailures import RainAlteration_1, Condensation_1


def pre_processing(img):
    """
    Pre-process the image for classification, in the same way of the pictures
    used to train the CNN
    """
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        img_float32 = np.float32(img)
        img = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img.astype(np.uint8))
    img = img/255
    img = np.expand_dims(img, axis=0)
    if not(len(np.shape(img)) == 3 and np.shape(img)[2] == 1):
        img = np.expand_dims(img, axis=-1)
    return img


if __name__ == '__main__':
    # load the model
    model = load_model('models\\TrafficSignModel.h5')

    # set the accuracy threshold
    accuracy_threshold = 0.8

    # get the images in the test data-set
    # X_test is the set of images, while y_test is the set of labels
    with open('models\\TestDataset.p', 'rb') as f:
        test_data = pickle.load(f)
        X_test, y_test = test_data['features'], test_data['labels']
        assert(X_test.shape[0] == y_test.shape[0]), \
            "Number of Images is not equals to number of labels"
        assert(X_test.shape[1:] == (32, 32, 3)), \
            "The dimensions of the images are not 32 x 32 x 3"

    # set the classes
    classes = list(range(44))
    classesStr = [str(classV) for classV in classes]

    # load the environment
    environment = EnvironmentRTest(model, X_test[:200], classesStr,
                                   preprocess_f=pre_processing,
                                   label_list=y_test[:200])

    # get the standard behavior of the net
    accuracy = classification(environment)

    # create the alteration_type as a RainAlteration1
    alteration_type: Alteration = RainAlteration_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Condensation1
    alteration_type = Condensation_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a GaussianNoise with variance 0.01
    alteration_type = GaussianNoise(0, 1, 0.01)

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

    # create the alteration_type as a Blur Variation with radius = 2, with
    # black and white images (L)
    alteration_type = Blur(0, 1, 1, 'L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)

    # create the alteration_type as a Brightness Variation, with black and
    # white images (L)
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
