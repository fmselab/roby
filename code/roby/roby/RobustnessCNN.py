"""
Created on Wed May 29 17:22:43 2019

@author: Andrea Bombarda
"""
import matplotlib.pyplot as plt   # type: ignore
import cv2   # type: ignore
# Manage csv files
import csv
from roby.RobustnessResults import RobustnessResults
from builtins import isinstance
from roby import Alterations, EnvironmentRTest
from typing import List


def set_classes(filename: str) -> List[str]:
    """
    Loads the classes from the CSV file and returns the list

    Parameters
    ----------
        filename : str
            the path of the csv file containing the classes definition

    Returns
    -------
        classes : list
            the list (of str) containing the name of the classes
    """
    classes = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            classes.append(row[0])
    return classes


def compute_robustness(accuracies: List[float], steps: List[float],
                       threshold: float) -> float:
    """
    Computes the robustness starting from the accuracies.
    It is computed counting the number of points with accuracy over the
    threshold, divided for the total number of points.

    Parameters
    ----------
        accuracies : list
            the list (of float) of the accuracy values gathered during the
            robustness analysis
        steps : list
            the list (of float) of the steps used to evaluate the robustness
        threshold : float
            the value chosen as the acceptable limit of accuracy to calculate
            the robustness

    Returns
    -------
        robustness : float
            the robustness computed for the CNN under analysis w.r.t. the given
            alteration

    """
    above_threshold = sum(i >= threshold for i in accuracies)
    return float(above_threshold)/float(len(steps))


def classification(environment: EnvironmentRTest.EnvironmentRTest) -> float:
    """
    Just a simple classification performed form the model we uploaded.
    This methods performs the classification using un-altered images and
    returns the accuracy of the network.

    Parameters
    ----------
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis

    Returns
    -------
        accuracy : float
            the accuracy of the CNN under analysis
    """
    successes = 0
    failures = 0
    image_index = 0
    for thisFile in environment.file_list:
        if isinstance(thisFile, str):
            img = cv2.imread(thisFile)
        else:
            img = thisFile
        # Pre-process the image for classification
        if environment.pre_processing is not None:
            img = environment.pre_processing(img)
        # Classify the input
        proba = environment.model.predict(img)[0]
        if environment.post_processing is not None:
            proba = environment.post_processing(proba)

        # Get predicted label and real one
        predicted_class = 0
        predicted_prob = 0
        for (label, p) in zip(environment.classes, proba):
            if float(p) > float(predicted_prob):
                predicted_class = label
                predicted_prob = p
        if environment.label_list is not None:
            real_label = environment.label_list[image_index]
        else:
            raise RuntimeError("Real lable list cannot be None")

        # Classify the type of the classification
        if str(predicted_class) == str(real_label):
            successes += 1
        else:
            failures += 1
        image_index = image_index + 1

    accuracy = float(successes) / float(environment.total_img)
    print('Successes: ' + str(successes))
    print('Failures: ' + str(failures))
    print('Accuracy: ' + str(accuracy))
    return accuracy


def display_robustness_results(results: RobustnessResults):
    """
    Display the results of robustness analysis.
    This methods print the robustness and creates a plot (which is then stored
    in a .jpg image) of the accuracy variation over different
    levels of alteration.

    Parameters
    ----------
        results : RobustnessResults
            the results of the robustness analysis
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(results.steps, results.accuracies)
    plt.title(results.title)
    plt.xlabel(results.xlabel)
    plt.ylabel(results.ylabel)
    plt.savefig(results.title + '.jpg')
    print('Robustness w.r.t ' + results.alteration_name + ': ' +
          str(results.robustness))


def robustness_test(environment: EnvironmentRTest.EnvironmentRTest,
                    alteration: Alterations.Alteration,
                    n_values: int,
                    accuracy_threshold: float) -> RobustnessResults:
    """
    Executes robustness analysis on a given alteration.

    Parameters
    ----------
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis
        alteration : Alteration
            the alteration w.r.t. the user wants to compute the robustness of
            the CNN
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
    accuracies = []
    for step in alteration.get_range(n_values):
        steps.append(step)
        # Reset the parameters to count
        successes = 0
        failures = 0
        image_index = 0
        for thisFile in environment.file_list:
            if isinstance(thisFile, str):
                img = alteration.apply_alteration(thisFile, step)
            else:
                img = alteration.apply_alteration_image(thisFile, step)
            # Pre-processing Function
            if environment.pre_processing is not None:
                img = environment.pre_processing(img)
            proba = environment.model.predict(img)[0]
            # Post-processing Function, the probability has to be in the same
            # order of the classes
            if environment.post_processing is not None:
                proba = environment.post_processing(proba)
            # Get predicted label and real one
            predicted_class = 0
            predicted_prob = 0
            for (label, p) in zip(environment.classes, proba):
                if float(p) > float(predicted_prob):
                    predicted_class = label
                    predicted_prob = p

            if environment.label_list is not None:
                real_label = environment.label_list[image_index]
            else:
                raise RuntimeError("Real lable list cannot be None")

            # Classify the type of the classification
            if str(predicted_class) == str(real_label):
                successes += 1
            else:
                failures += 1
            image_index = image_index + 1

        # All of the images have been processed, so we can compute the accuracy
        # for this step value
        accuracy = float(successes) / float(environment.total_img)
        accuracies.append(accuracy)

    # Plot data
    title = 'Accuracy over ' + alteration.name() + ' Alteration'
    xlabel = 'Image Alteration - ' + alteration.name()
    ylabel = 'Accuracy'

    # Robustness computation
    robustness = compute_robustness(accuracies, steps, accuracy_threshold)
    results = RobustnessResults(steps, accuracies, robustness, title, xlabel,
                                ylabel, alteration.name())
    return results
