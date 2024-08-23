"""
In this **BDD100k identification** example we test the robustness of the
two NNs, i.e., the original one and the one we repaired usng eAI-Repair-Toolkit
"""
from keras.models import load_model   # type: ignore
from roby.RobustnessNN import robustness_test, set_classes,\
    display_robustness_results, classification
from roby.Alterations import GaussianNoise
from roby.EnvironmentRTest import EnvironmentRTest
from imutils import paths   # type: ignore
import imutils
import h5py
from matplotlib import pyplot as plt
from cameraFailures import RainAlteration_1, Condensation_1, Ice_1
import cv2   # type: ignore
from keras.preprocessing.image import img_to_array   # type: ignore
import numpy as np   # type: ignore
from roby.Alterations import Blur, Brightness, Alteration
    


def reader(file_name):
    return file_name


def pre_processing(image):
    """
    Pre-processes the image for classification, in the same way of the pictures
    used to train the CNN
    """
    imutils.resize(image, width=32)
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def labeler(image):
    """
    Extracts the label from the image file name
    """
    real_label = (image.split('.')[0]).split('_')[-1]
    return real_label


class H5_data():
    def __init__(self, path): 
        self.path = path
        self.image, self.label = H5_data.read_h5(path)

    @classmethod
    def read_h5(cls,path):
        with h5py.File(path, 'r') as data:
            image = data['images'][()]
            label = data['labels'][()]
            return image, label


if __name__ == '__main__':
    # load the model
    model = load_model('model/originalModel.h5')
    # set the accuracy threshold
    accuracy_treshold = 0.8
    # load images and labels
    h5_test = H5_data('images/test.h5')

    # Lettura immagini OK
    # plt.imshow(h5_test.image[100])
    # plt.show() 

    # labels
    # print(set(map(lambda x: str(x), h5_test.label)))

    
    # get the images in the test data-set
    file_list = h5_test.image

    # set the classes
    classes = set_classes('model/Classes.csv')

    # load the environment
    environment = EnvironmentRTest(model, file_list[:1000], classes,
                                   label_list = h5_test.label[:1000],
                                   preprocess_f=pre_processing,
                                   reader_f=reader)

    # get the standard behavior of the net
    accuracy = classification(environment)


    """
    # create the alteration_type as a GaussianNoise with variance 200
    alteration_type: Alteration = GaussianNoise(0, 1, 200)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Blur Variation, with radius = 2
    alteration_type = Blur(0, 1, 2)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Brightness Variation
    alteration_type = Brightness(-0.5, 0.5, "L")

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a RainAlteration1
    alteration_type: Alteration = RainAlteration_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Condensation1
    alteration_type = Condensation_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Ice1
    alteration_type = Ice_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    """
