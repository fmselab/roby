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
from roby.RobustnessCNN import classification, robustness_test,\
    display_robustness_results
from roby.Alterations import Alteration
from keras.engine.training import Model
from utils import get_data
from audioFailures import AudioNoise


def pre_processing(data):
    """
    Pre-processes the image for classification, in the same way of the pictures
    used to train the CNN
    """
    data_processed = np.expand_dims(data, axis=0)
    return data_processed


if __name__ == '__main__':
    # load the model
    model = load_model('models\\trained_model.h5')

    # set the accuracy threshold
    accuracy_threshold = 0.8

    # get the audio files
    X_train, x_test, y_train, y_test, cnn_model = get_data.get_all()

    # set the classes
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # load the environment
    environment = EnvironmentRTest(model, x_test, classes,
                                   label_list=y_test, preprocess_f=pre_processing)

    # get the standard behavior of the net
    accuracy = classification(environment)
    
    # create the alteration_type as a AudioNoise
    alteration_type: Alteration = AudioNoise(0, 1, 200)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_threshold)
    display_robustness_results(results)