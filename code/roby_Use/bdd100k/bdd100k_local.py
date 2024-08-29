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
from roby.Alterations import GaussianNoise, Alteration
from roby.EnvironmentRTest import EnvironmentRTest
import h5py
from matplotlib import pyplot as plt
from cameraFailures import RainAlteration_1, Condensation_1, Ice_1, CustomBrightness, CustomBlur
import numpy as np   # type: ignore
from typing import List, Tuple
from datetime import datetime
from roby.RobustnessResults import RobustnessResults
    

def reader(file_name: np.ndarray) -> np.ndarray:
    """
    A trivial reader function. The images are already in the np.ndarray format,
    so we do not need to read them.

    Parameters
    ----------
        file_name : np.ndarray
            the input image

    Returns
    -------
        file_name : np.ndarray
            the read image
    """
    return file_name


def _parse_test_results(test_images, test_labels, results):
    """
    Parse test results.
    The function has been taken from the eAI-Repair-Toolkit repository.

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
    results = model.predict(input_set, verbose=1, batch_size=1000)
    successes, _ = _parse_test_results(input_set, environment.label_list, results)
    n_successes = 0
    # Count the number of successes
    for key in successes.keys():
            n_successes += len(successes[key])
    # Compute the accuracy
    accuracy = n_successes / input_set.__len__()
    return accuracy


def robustness_test_batch(environment: EnvironmentRTest,
                    alteration: Alteration,
                    n_values: int,
                    accuracy_threshold: float) -> RobustnessResults:
    """
    Executes robustness analysis on a given alteration.
    It uses a batch classification in order to speed up the robustness computation.

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
    """
    Class to read the data from a ".h5" file.

    Attributes
    ----------
        image : np.ndarray
            the images contained in the ".h5" file
        label : np.ndarray
            the labels contained in the ".h5" file
    """

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
    # parameter setting
    model_name = 'model/repairedModel.h5'
        # 'model/originalModel.h5'
    input_set = 'images/test.h5'
    classes_file = 'model/Classes.csv'
    label = "[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]"
    accuracy_treshold = 0.7
    evaluate_network = False
    evaluate_robustness = True
    show_preview = True
    alterations = ["RA"]
        #["GN", "BL", "BR", "RA", "CO", "IC"]

    # load the model
    model = load_model(model_name)
    
    # load images and labels
    h5_test = H5_data(input_set)

    # set the classes
    classes = set_classes(classes_file)

    # Extract from the test set only images apartaining to the pedestrian category
    input_set = []
    labels = []
    if label is not None:
        for i in range(len(h5_test.label)):
            if str(h5_test.label[i]) == label:
                input_set.append(h5_test.image[i])
                labels.append(h5_test.label[i])
    else:
        input_set = h5_test.image
        labels = h5_test.label
  
    # load the environment
    environment = EnvironmentRTest(model, input_set, classes,
                                   label_list = labels,
                                   reader_f=reader)
    
    if evaluate_network:
        # get the standard behavior of the net
        print("Evaluating network performance on nominal data")
        accuracy = batch_classification(environment, np.array(input_set))
        print("Accuracy: ", accuracy)

    if evaluate_robustness:
        # evaluate all alterations
        for alt in alterations:
            # depending on the alteration, we need to create the alteration_type
            alteration_type = None
            if alt == "GN":
                alteration_type = GaussianNoise(0, 0.05, 5)
            elif alt == "BL":
                alteration_type = CustomBlur(0, 0.5, 1, "float64")
            elif alt == "BR":
                alteration_type = CustomBrightness(-1, 1, "float64")
            elif alt == "RA":
                alteration_type = RainAlteration_1('L')
            elif alt == "CO":
                alteration_type = Condensation_1('L')
            elif alt == "IC":
                alteration_type = Ice_1('L')
            else:
                print("Unknown alteration")
                continue

            if show_preview:
                # show the preview of the alteration
                f, axarr = plt.subplots(3,1) 
                axarr[0].imshow(alteration_type.apply_alteration(h5_test.image[0], alteration_type.value_from))
                axarr[1].imshow(alteration_type.apply_alteration(h5_test.image[0], (alteration_type.value_from + alteration_type.value_to) / 2))
                axarr[2].imshow(alteration_type.apply_alteration(h5_test.image[0], alteration_type.value_to))
                plt.show()

            # perform robustness analysis, with 20 points
            results = robustness_test_batch(environment, alteration_type, 20,
                                    accuracy_treshold)
            display_robustness_results(results)