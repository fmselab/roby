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
from PIL import ImageShow, Image

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



if __name__ == '__main__':
    # load the model
    model = load_model('model/Medical.model')
    # set the accuracy threshold
    accuracy_treshold = 0.8
    # get the images in the test data-set
    file_list = sorted(list(paths.list_images('images')))[:10]
    # classes
    classes = ["class0", "class1"]
    # reader
    reader = lambda file_name: cv2.imread(file_name)
    # labeler
    labeler = lambda image: (image.split('.')[0]).split('_')[-1]
    
    # load the environment
    environment = EnvironmentRTest(model, file_list, classes,
                                   labeler_f=labeler,
                                   preprocess_f=pre_processing,
                                   reader_f=reader)

    # create the alteration_type as a Horizontal Translation
    alteration_type = HorizontalTranslation(-1, 1)

    # perform robustness analysis, with 200 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    im = Image.open('Accuracy over HorizontalTranslation Alteration.jpg')
    im.show()
    
