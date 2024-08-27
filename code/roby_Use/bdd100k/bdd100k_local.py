"""
In this **BDD100k identification** example we test the robustness of the
two NNs, i.e., the original one and the one we repaired usng eAI-Repair-Toolkit

The test dataset is contained into a ".h5" file, which is loaded and used to
test the robustness of the two NNs. Input pictures are stored into a np.ndarray
with shape (n, 32, 32, 3), where n is the number of pictures in the dataset, and
each element is float64.
"""
from keras.models import load_model   # type: ignore
from roby.RobustnessNN import set_classes,\
    display_robustness_results, compute_robustness
from roby.Alterations import GaussianNoise
from roby.EnvironmentRTest import EnvironmentRTest
import imutils
import h5py
from matplotlib import pyplot as plt
from cameraFailures import RainAlteration_1, Condensation_1, Ice_1, CustomBrightness, CustomBlur
import cv2   # type: ignore
import numpy as np   # type: ignore
from typing import List, Tuple
from datetime import datetime
from roby.RobustnessResults import RobustnessResults
    


def reader(file_name):
    return file_name

def _parse_test_results(test_images, test_labels, results):
    """Parse test results.

    Parse test results and split them into success and failure datasets.
    Both datasets are dict of list consisted of dict of image and label.
    successes: {0: [{'image': test_image, 'label': test_label}, ...],
                1: ...}
    failures: {0: {1: [{'image': test_image, 'label': test_label}, ...],
                3: [{'image': test_iamge, 'label': test_label}, ...],
                ...},
            1: {0: [{'image': test_image, 'label': test_label}, ...],
                2: [{'image': test_iamge, 'label': test_label}, ...],
                ...}}

    Parameters
    ----------
    test_images :
        Images used to test moel
    test_labels :
        Labels used to test model
    results : list
        Results of prediction

    Returns
    -------
    successed, failure
        Results of successes and failures

    """
    successes = {}
    failures = {}
    dataset_len = len(test_labels)
    for i in range(dataset_len):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()

        result = results[i]
        predicted_label = result.argmax()
        if predicted_label != test_label_index:
            if test_label_index not in failures:
                failures[test_label_index] = {}
            if predicted_label not in failures[test_label_index]:
                failures[test_label_index][predicted_label] = []
            failures[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label}
            )
        else:
            if test_label_index not in successes:
                successes[test_label_index] = []
            successes[test_label_index].append({"image": test_image, "label": test_label})
    return successes, failures


def batch_classification(environment: EnvironmentRTest, input_set: np.ndarray) -> float:
    """
    Performs a batch classification on a test set with a given model and returns
    the measured accuracy. It uses a batch size of 1000.

    Parameters
    ----------
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis
        input_set : np.ndarray
            the input set. We do not use that in the environment, since we could
            apply the alteration to the input set

    Returns
    -------
        accuracy : float
            the accuracy computed with the given model
    """
    successes = 0
    # Predict the test set using a batch size of 1000
    results = model.predict(file_list, verbose=1, batch_size=1000)
    successes, failures = _parse_test_results(input_set, environment.label_list, results)
    n_successes = 0
    # Count the number of successes
    for key in successes.keys():
            n_successes += len(successes[key])
    # Compute the accuracy
    accuracy = n_successes / input_set.__len__()
    return accuracy

def robustness_test_batch(environment,
                    alteration,
                    n_values,
                    accuracy_threshold):
    """
    Executes robustness analysis on a given alteration.

    Parameters
    ----------
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis
        alteration : Alteration
            the alteration w.r.t. the user wants to compute the robustness of
            the NN
        n_values : int
            the number of points in the interval to be used for robustness
            analysis
        accuracy_threshold : float
            acceptable limit of accuracy to calculate the robustness

    Returns
    -------
        results : RobustnessResults
            the results of the robustness test
    """
    assert 0.0 <= accuracy_threshold <= 1.0
    steps = []
    for step in alteration.get_range(n_values):
        steps.append(step)
    accuracies = []
    successes = []
    failures = []
    times = []

    print ("[" + str(datetime.now()) + "]: Starting alteration " +
           alteration.name())

    for step_index in range(0, len(steps)):
        successes.append(0)
        failures.append(0)
        times.append(0.0)

        step = steps[step_index]   
        file_list = [alteration.apply_alteration(x, step) for x in environment.file_list]
        file_list = np.array(file_list)

        accuracy = batch_classification(environment, file_list)
        accuracies.append(accuracy)
        
    # Plot data
    title = 'Accuracy over ' + alteration.name() + ' Alteration'
    xlabel = 'Image Alteration - ' + alteration.name()
    ylabel = 'Accuracy'

    print ("[" + str(datetime.now()) + "]: Ending alteration " +
           alteration.name())
    
    # Robustness computation
    robustness = compute_robustness(accuracies, steps, accuracy_threshold)
    results = RobustnessResults(steps, accuracies, robustness, title, xlabel,
                                ylabel, alteration.name(), accuracy_threshold,
                                times)
    return results


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
    model = load_model('model/repairedModel.h5')
    # set the accuracy threshold
    accuracy_treshold = 0.8
    # load images and labels
    h5_test = H5_data('images/test.h5')

    # set the classes
    classes = set_classes('model/Classes.csv')
  
    # load the environment
    environment = EnvironmentRTest(model, h5_test.image, classes,
                                   label_list = h5_test.label,
                                   reader_f=reader)
    
    # get the standard behavior of the net
    # print (batch_classification(environment, h5_test.image))

    # get the standard behavior of the net
    # accuracy = classification(environment)
    
    """
    # create the alteration_type as a GaussianNoise with variance 20
    alteration_type: Alteration = GaussianNoise(0, 0.05, 20)

    # perform robustness analysis, with maximum 20 points
    results = robustness_test_batch(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    """
    
    """
    # create the alteration_type as a Blur Variation, with radius = 2
    alteration_type = CustomBlur(0, 0.5, 1, "float64")

    # perform robustness analysis, with 20 points
    results = robustness_test_batch(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    """   
    
    """
    # create the alteration_type as a Brightness Variation
    alteration_type = CustomBrightness(-1, 1, "float64")

    # perform robustness analysis, with 20 points
    results = robustness_test_batch(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    """
    
    """
    # create the alteration_type as a RainAlteration1
    alteration_type: Alteration = RainAlteration_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    """
    
    # create the alteration_type as a Condensation1
    alteration_type = Condensation_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test_batch(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)

    """
    # create the alteration_type as a Ice1
    alteration_type = Ice_1('L')

    # perform robustness analysis, with 20 points
    results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
    display_robustness_results(results)
    """
