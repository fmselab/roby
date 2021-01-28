"""
The **RobustnessNN** module offers the main functionalities of the
**roby** tool, i.e. those useful to compute the robustness and evaluate
a neural network.

The major features offered by this module are the followings:

- Network evaluation via its accuracy
- Robustness evaluation, applying a specific alteration
- Robustness results display

@author: Andrea Bombarda
"""
import matplotlib.pyplot as plt   # type: ignore
# Manage csv files
import csv
import numpy as np  # type: ignore
from roby.RobustnessResults import RobustnessResults
from roby import Utility
from builtins import isinstance
from roby import Alterations, EnvironmentRTest
from typing import List, Tuple
from datetime import datetime
import math


def set_classes(filename: str) -> List[str]:
    """
    Loads the classes from the CSV file and returns the list

    Parameters
    ----------
        filename : str
            the path of the csv file containing the classes definition

    Returns
    -------
        classes : List[str]
            the list (of str) containing the name of the classes
    """
    classes = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            classes.append(row[0])
    return classes


def compute_robustness(accuracies: List[float], steps: List[float],
                       threshold: float, lower_bound: float = None,
                       upper_bound: float = None) -> float:
    """
    Computes the robustness starting from the accuracies.
    It is computed counting the number of points with accuracy over the
    threshold, divided for the total number of points.
    If the two bounds are given, the robustness is weighted by the step
    lenghts

    Parameters
    ----------
        accuracies : List[float]
            the list (of float) of the accuracy values gathered during the
            robustness analysis
        steps : List[float]
            the list (of float) of the steps used to evaluate the robustness
        threshold : float
            the value chosen as the acceptable limit of accuracy to calculate
            the robustness
        lower_bound : float, optional
            the value of the lower bound for robustness computation
        upper_bound : float, optional
            the value of the upper bound for robustness computation

    Returns
    -------
        robustness : float
            the robustness computed for the NN under analysis w.r.t. the given
            alteration

    """
    if upper_bound is None and lower_bound is None:
        above_threshold = sum(i >= threshold for i in accuracies)
        return float(above_threshold)/float(len(steps))
    elif upper_bound is not None and lower_bound is not None:
        i = 0
        above_threshold_2 = 0.0
        for a in accuracies:
            above_threshold_2 = above_threshold_2 + (
                (1.0 if a >= threshold else 0.0) * (
                    (steps[i]-steps[i-1]) if i > 0 else 0.0))
            i = i+1
        return float(above_threshold_2)/float(upper_bound - lower_bound)
    else:
        raise RuntimeError("Unable to compute robustness without bounds")


def classification(environment: EnvironmentRTest.EnvironmentRTest) -> float:
    """
    Just a simple classification performed form the model we uploaded.
    This methods performs the classification using un-altered input data and
    returns the accuracy of the network.

    Parameters
    ----------
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis

    Returns
    -------
        accuracy : float
            the accuracy of the NN under analysis
    """
    successes = 0
    failures = 0
    data_index = 0
    for thisFile in environment.file_list:
        if isinstance(thisFile, str):
            if environment.reader_f is None:
                raise RuntimeError("A reader function must be defined")
            imgt = environment.reader_f(thisFile)
        else:
            imgt = thisFile
        # Pre-process the input data for classification
        if environment.pre_processing is not None:
            data = environment.pre_processing(imgt)
        else:
            data = imgt
        # Classify the input
        proba = environment.model.predict(data)[0]

        # Get predicted label and real one
        predicted_class = ""
        predicted_prob = 0
        for (label, p) in zip(environment.classes, proba):
            if float(p) > float(predicted_prob):
                predicted_class = str(label)
                predicted_prob = p
        if environment.label_list is not None:
            real_label = environment.label_list[data_index]
        else:
            raise RuntimeError("Real lable list cannot be None")

        # Classify the type of the classification
        if str(predicted_class) == str(real_label):
            successes += 1
        else:
            failures += 1
        data_index = data_index + 1

    accuracy = float(successes) / float(environment.total_data)
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
    thresholds = np.full(len(results.steps), results.threshold)
    plt.plot(results.steps, thresholds)
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
    times: List[float] = []
    data_index = 0

    print ("[" + str(datetime.now()) + "]: Starting alteration " +
           alteration.name())

    for step_index in range(0, len(steps)):
        successes.append(0)
        failures.append(0)
        times.append(0.0)

    for thisFile in environment.file_list:
        inputFile = thisFile
        start_loading_time = datetime.now()
        if isinstance(inputFile, str):
            if environment.reader_f is None:
                raise RuntimeError("A reader function must be defined")
            inputFile = environment.reader_f(thisFile)
        end_loading_time = datetime.now()
        total_loading_time = \
            (end_loading_time - start_loading_time).total_seconds()

        for step_index in range(0, len(steps)):
            start_time = datetime.now()
            step = steps[step_index]
            data = alteration.apply_alteration(inputFile, step)
            # Pre-processing Function
            if environment.pre_processing is not None:
                data = environment.pre_processing(data)
            proba = environment.model.predict(data)[0]
            # Get predicted label and real one
            predicted_class = ""
            predicted_prob = 0
            for (label, p) in zip(environment.classes, proba):
                if float(p) > float(predicted_prob):
                    predicted_class = str(label)
                    predicted_prob = p

            if environment.label_list is not None:
                real_label = environment.label_list[data_index]
            else:
                raise RuntimeError("Real lable list cannot be None")

            # Classify the type of the classification
            if str(predicted_class) == str(real_label):
                successes[step_index] += 1
            else:
                failures[step_index] += 1

            # Record the time
            end_time = datetime.now()
            times[step_index] = times[step_index] + \
                (end_time - start_time).total_seconds() + \
                total_loading_time

        data_index = data_index + 1

    for step_index in range(0, len(steps)):
        # All of the data have been processed, so we can compute the accuracy
        # for this step value
        accuracy = float(successes[step_index]) / float(environment.total_data)
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


def eval_parabola(x1: float, x2: float, threshold: float, minstep: float,
                  environment: EnvironmentRTest.EnvironmentRTest,
                  alteration: Alterations.Alteration,
                  a_threshold: float, ptype: str,
                  result_list: List[Tuple[float, float, float]]):
    """
    Recursively evaluates the accuracy depending on the parabola approximation

    Parameters
    ----------
        x1 : float
            the lower bound
        x2 : float
            the upper bound.
        threshold : float
            acceptable limit of accuracy to calculate the robustness
        minstep : float
            the minumum distance between points
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis
        alteration : Alteration
            the alteration w.r.t. the user wants to compute the robustness of
            the NN
        a_threshold : float
            the concavity of the parabola used for robustness approximation
        type : str
            the type of the parabola to be used for approximation. Use "real"
            for a real parabola, or "appr" for an approximated parabola.
            The former gives better results, in terms of accuracy, while the
            latter guarantees better performances. However, the difference
            in terms of accuracy between the two types is very low
        result_list : List[Tuple[float, float, float]]
            the list of entries [x, accuracy, time]
    """
    assert 0.0 <= threshold <= 1.0
    assert (ptype == "real" or ptype == "appr")

    # Check if points are too close
    if (x2 - x1 < minstep):
        return

    # Computes the two accuracies
    x1 = float(x1)
    x2 = float(x2)
    y1, time1 = get_accuracy(environment, alteration, x1)
    y2, time2 = get_accuracy(environment, alteration, x2)

    # Check whether the two points are already in the dictionary or not
    if len(list(filter(lambda a: a[0] == x1, result_list))) == 0:
        result_list.append((x1, y1, time1))
    if len(list(filter(lambda a: a[0] == x2, result_list))) == 0:
        result_list.append((x2, y2, time2))

    # If a point is above the threshold and the other is under
    if ((y1-threshold) * (y2-threshold) <= 0):
        eval_parabola(x1, (x2+x1)/2, threshold, minstep, environment,
                      alteration, a_threshold, ptype, result_list)
        eval_parabola((x2+x1)/2, x2, threshold, minstep, environment,
                      alteration, a_threshold, ptype, result_list)
        return

    if ptype == "real":
        # If the approximation type is "real", then calculate the parabola
        # using the real coordinates of the vertex
        params = Utility.compute_parabola(x1, y1, x2, y2, threshold)
        a = None
        for p in params:
            if (p[0] != 0):
                a = math.fabs(p[0])

        if a is None:
            raise RuntimeError("No parabola approximation is possible using" +
                               " these points")
    else:
        # If the approximation type is "real", then calculate the parabola
        # using the mean value between x1 and x2 for the vertex
        y3 = threshold
        x3 = (x1 + x2) / 2
        params = Utility.compute_appoximate_parabola(x1, y1, x2, y2, x3, y3)
        a = math.fabs(params[0])

    # if the "a" value is greater than the threshold, terminates the recursion
    if (a > a_threshold):
        return
    else:
        eval_parabola(x1, (x2+x1)/2, threshold, minstep, environment,
                      alteration, a_threshold, ptype, result_list)
        eval_parabola((x2+x1)/2, x2, threshold, minstep, environment,
                      alteration, a_threshold, ptype, result_list)
        return


def approximate_robustness_test(environment: EnvironmentRTest.EnvironmentRTest,
                                alteration: Alterations.Alteration,
                                n_values: int,
                                accuracy_threshold: float,
                                a: float,
                                ptype: str) -> RobustnessResults:
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
        a : float
            the concavity of the parabola used for robustness approximation
        ptype : str
            the type of the parabola to be used for approximation. Use "real"
            for a real parabola, or "appr" for an approximated parabola.
            The former gives better results, in terms of accuracy, while the
            latter guarantees better performances. However, the difference
            in terms of accuracy between the two types is very low

    Returns
    -------
        results : RobustnessResults
            the results of the robustness test
    """
    assert 0.0 <= accuracy_threshold <= 1.0
    assert (ptype == "real" or ptype == "appr")

    print ("[" + str(datetime.now()) + "]: Starting alteration " +
           alteration.name())

    lower_bound = alteration.get_range(n_values).min()
    upper_bound = alteration.get_range(n_values).max()
    minstep = float(upper_bound - lower_bound) / float(n_values)
    result_list: List[Tuple[float, float, float]] = []
    partial_results: List[Tuple[str, np.ndarray, float, str, float]] = []

    eval_parabola(lower_bound, upper_bound, accuracy_threshold, minstep,
                  environment, alteration, a, ptype,
                  result_list)

    # Sort the list by key (x values)
    result_list = sorted(result_list, key=lambda i: i[0])
    partial_results = sorted(partial_results, key=lambda i: i[1])
    steps = list(elem[0] for elem in result_list)
    accuracies = list(elem[1] for elem in result_list)
    times = list(elem[1] for elem in result_list)

    # Plot data
    title = 'Accuracy over ' + alteration.name() + ' Alteration'
    xlabel = 'Image Alteration - ' + alteration.name()
    ylabel = 'Accuracy'

    print ("[" + str(datetime.now()) + "]: Ending alteration " +
           alteration.name())

    # Robustness computation
    robustness = compute_robustness(accuracies, steps, accuracy_threshold,
                                    lower_bound, upper_bound)
    results = RobustnessResults(steps, accuracies, robustness, title, xlabel,
                                ylabel, alteration.name(), accuracy_threshold,
                                times)
    return results


def get_accuracy(environment: EnvironmentRTest.EnvironmentRTest,
                 alteration: Alterations.Alteration,
                 level: float) -> Tuple[float, float]:
    """
    Compute the accuracy of the model when an alteration of a defined level is
    applied

    Parameters
    ----------
        environment : EnvironmentRTest
            the environment containing all the information used to perform
            robustness analysis
        alteration : Alteration
            the alteration w.r.t. the user wants to compute the robustness of
            the NN
        level : float
            the level of the alteration to be applied

    Returns
    -------
        accuracy, time : float, float
            the accuracy of the network and the time required for applying
            the desired alteration level
    """
    successes = 0
    failures = 0
    data_index = 0
    start_time = datetime.now()

    # Fetch all the files
    for thisFile in environment.file_list:
        inputFile = thisFile
        if isinstance(inputFile, str):
            if environment.reader_f is None:
                raise RuntimeError("A reader function must be defined")
            inputFile = environment.reader_f(thisFile)
        # Apply the alteration
        data = alteration.apply_alteration(inputFile, level)
        # Pre-processing Function
        if environment.pre_processing is not None:
            data = environment.pre_processing(data)
        proba = environment.model.predict(data)[0]
        # Get predicted label and real one
        predicted_class = ""
        predicted_prob = 0
        for (label, p) in zip(environment.classes, proba):
            if float(p) > float(predicted_prob):
                predicted_class = str(label)
                predicted_prob = p

        # Real label
        if environment.label_list is not None:
            real_label = environment.label_list[data_index]
        else:
            raise RuntimeError("Real lable list cannot be None")

        # Classify the type of the classification
        if str(predicted_class) == str(real_label):
            successes += 1
        else:
            failures += 1

        data_index = data_index + 1

    end_time = datetime.now()

    return float(successes) / float(data_index),\
        (end_time-start_time).total_seconds()
