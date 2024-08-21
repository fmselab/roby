"""
The script demonstrates a simple example of using ART with Keras.
The example train a model on the CelebA dataset
and creates adversarial examples using the Fast Gradient Sign Method.
"""
import numpy as np  # type: ignore
from keras.applications.inception_v3 import InceptionV3  # type: ignore
from keras.layers.pooling import GlobalAveragePooling2D  # type: ignore
from keras.layers.core import Dense, Dropout  # type: ignore
from keras.engine.training import Model  # type: ignore
import pandas as pd  # type: ignore

from art.attacks.evasion import FastGradientMethod  # type: ignore
from art.estimators.classification import KerasClassifier  # type: ignore

from imutils import paths   # type: ignore
import cv2   # type: ignore
from roby.EnvironmentRTest import EnvironmentRTest

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


img_width = 178
img_height = 218
df = pd.read_csv("/mnt/e/Labels/CelebA/list_attr_celeba.txt", header=0,
                 delimiter=r"\s+")


def reader(file_name):
    image = cv2.imread(file_name)
    image = image.astype("float") / 255.0
    return image


def labeler(image):
    """
    Search the class in the class file
    """
    fileName = image.split('/')
    df2 = df[df["file"] == fileName[-1]]

    if str(df2.iloc[0]['Male']) == "-1":
        return 0
    else:
        return 1


def load_model():
    model = InceptionV3(weights='/mnt/e/Models/CelebA/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False,
                        input_shape=(img_height, img_width, 3))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    predictionsM = Dense(2, activation='softmax')(x)
    model_ = Model(inputs=model.input, outputs=predictionsM)
    model_.load_weights('/mnt/e/Models/CelebA/weights.best.inc.male.hdf5')
    return model_


if __name__ == '__main__':
    # load the model
    model = load_model()

    # get the images in the test data-set
    file_list = sorted(list(paths.list_images('/mnt/e/Datasets/CelebA')))

    # set the classes
    classes = ["Female", "Male"]

    # load the environment
    environment = EnvironmentRTest(model, file_list, classes,
                                   labeler_f=labeler)

    assert environment.label_list is not None
    assert environment.file_list is not None

    # Create the ART classifier for Keras models
    classifier = KerasClassifier(model=model, clip_values=(0, 1),
                                 use_logits=False)

    # Train the ART classifier
    classifier.fit(list(map(reader, environment.file_list[:16000])),
                   environment.label_list[:16000], batch_size=64, nb_epochs=3)

    # Generate adversarial test examples
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=list(map(reader,
                                            environment.file_list[16001:])))

    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) ==
                      np.argmax(
                          environment.label_list[16001:],
                          axis=1)) / len(
                              environment.label_list[16001:])

    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
