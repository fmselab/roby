from roby.RobustnessNN import classification, robustness_test,\
    display_robustness_results
from roby.EnvironmentRTest import EnvironmentRTest
from imutils import paths   # type: ignore
import imutils
import cv2   # type: ignore
from keras.preprocessing.image import img_to_array   # type: ignore
import numpy as np   # type: ignore
import pandas as pd  # type: ignore
from keras.applications.inception_v3 import InceptionV3  # type: ignore
from keras.layers.pooling import GlobalAveragePooling2D  # type: ignore
from keras.layers.core import Dense, Dropout  # type: ignore
from keras.engine.training import Model  # type: ignore
from roby.Alterations import Alteration, GaussianNoise, Compression,\
    VerticalTranslation, HorizontalTranslation, Blur, Brightness, Zoom


img_width = 178
img_height = 218
df = pd.read_csv("E:/Labels/CelebA/list_attr_celeba.txt", header=0,
                 delimiter=r"\s+")


def reader(file_name):
    return cv2.imread(file_name)


def pre_processing(image):
    """
    Pre-processes the image for classification, in the same way of the pictures
    used to train the CNN
    """
    imutils.resize(image, width=img_width)
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def labeler(image):
    """
    Search the class in the class file
    """
    fileName = image.split('\\')
    df2 = df[df["file"] == fileName[1]]

    if str(df2.iloc[0]['Male']) == "-1":
        return "Female"
    else:
        return "Male"


def load_model():
    model = InceptionV3(
        weights='E:/Models/CelebA/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False,
        input_shape=(img_height, img_width, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model_ = Model(inputs=model.input, outputs=predictions)
    model_.load_weights('E:/Models/CelebA/weights.best.inc.male.hdf5')
    return model_


if __name__ == '__main__':

    # load the model
    model = load_model()

    # set the accuracy threshold
    accuracy_treshold = 0.8
    # get the images in the test data-set
    file_list = sorted(list(paths.list_images('E:/Datasets/CelebA')))

    # set the classes
    classes = ["Female", "Male"]

    # load the environment
    environment = EnvironmentRTest(model, file_list, classes,
                                   preprocess_f=pre_processing,
                                   labeler_f=labeler,
                                   reader_f=reader)

    # get the standard behavior of the net
    accuracy = classification(environment)

    # create the alteration_type as a GaussianNoise with variance 200
    alteration_type: Alteration = GaussianNoise(0, 1, 200)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    # create the alteration_type as a Compression
    alteration_type = Compression(0, 1)

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
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
