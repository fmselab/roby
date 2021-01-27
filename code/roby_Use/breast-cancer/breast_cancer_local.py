"""
In this **Breast Cancer identification** example we test the robustness of the
CNN we presented in
[this paper](https://ieeexplore.ieee.org/abstract/document/9176802).

In this example:

- The _images_ are loaded as a list of paths, because they are included into
  the `images` folder.
- The _classes_ are loaded from a `csv` file.
- The _labels_ are assigned using a labeler function. In our case study all
  the images contains `class0` or `class1` in the name of the file.
- Only _alterations_ plausible for the application domain are applied.
"""
from keras.models import load_model   # type: ignore
from roby.RobustnessNN import robustness_test, set_classes,\
    display_robustness_results, classification,\
    approximate_robustness_test
from roby.Alterations import GaussianNoise
from roby.EnvironmentRTest import EnvironmentRTest
from imutils import paths   # type: ignore
import imutils
import cv2   # type: ignore
from keras.preprocessing.image import img_to_array   # type: ignore
import numpy as np   # type: ignore
from roby.Alterations import Compression, VerticalTranslation,\
    HorizontalTranslation, Blur, Zoom, Brightness, Alteration,\
    AlterationSequence


def reader(file_name):
    return cv2.imread(file_name)


def pre_processing(image):
    """
    Pre-processes the image for classification, in the same way of the pictures
    used to train the CNN
    """
    imutils.resize(image, width=50)
    image = cv2.resize(image, (50, 50))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def labeler(image):
    """
    Extracts the label from the image file name
    """
    real_label = (image.split('.')[0]).split('_')[-1]
    return real_label


if __name__ == '__main__':
    # load the model
    model = load_model('model/Medical.model')
    # set the accuracy threshold
    accuracy_treshold = 0.8
    # get the images in the test data-set
    file_list = sorted(list(paths.list_images('images')))

    # set the classes
    classes = set_classes('model/Classes.csv')

    # load the environment
    environment = EnvironmentRTest(model, file_list, classes,
                                   preprocess_f=pre_processing,
                                   labeler_f=labeler,
                                   reader_f=reader)

    # get the standard behavior of the net
    accuracy = classification(environment)

    # create the alteration_type as a GaussianNoise with variance 200
    alteration_type: Alteration = GaussianNoise(0, 1, 200)

    # perform robustness analysis, with 200 points
    results = robustness_test(environment, alteration_type, 200,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Compression
    alteration_type = Compression(0, 1)

    # perform robustness analysis, with 200 points
    results = robustness_test(environment, alteration_type, 200,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Vertical Translation
    alteration_type = VerticalTranslation(-1, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Horizontal Translation
    alteration_type = HorizontalTranslation(-1, 1)

    # perform robustness analysis, with 200 points
    results = robustness_test(environment, alteration_type, 200,
                              accuracy_treshold)
    display_robustness_results(results)

    # perform approximate robustness analysis, with 200 points
    results = approximate_robustness_test(environment, alteration_type, 200,
                                          accuracy_treshold, 2, "real")
    display_robustness_results(results)

    # perform approximate robustness analysis, with 200 points
    results = approximate_robustness_test(environment, alteration_type, 200,
                                          accuracy_treshold, 2, "appr")
    display_robustness_results(results)

    # create the alteration_type as a Blur Variation, with radius = 2
    alteration_type = Blur(0, 1, 2)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Brightness Variation
    alteration_type = Brightness(-0.5, 0.5)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Zoom
    alteration_type = Zoom(0, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Sequence of Zoom and Brightness
    altseq = AlterationSequence([Zoom(0, 1), Brightness(-0.5, 0.5)])

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, altseq, 20, accuracy_treshold)
    display_robustness_results(results)
