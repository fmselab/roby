"""
The **RobustnessResults** module contains the class which is used to store
the results of each robustness analysis.

In particular, the following information are stored:

- List of the steps used for robustness analysis
- List of the accuracy values gathered during robustness analysis
- The value of the robustness
- The title of the plot, with the labels to be put on x and y axis
- The name of the applied alteration

@author: Andrea Bombarda
"""
from typing import List


class RobustnessResults:
    """Class storing the results of the robustness."""

    def __init__(self, steps: List[float], accuracies: List[float],
                 robustness: float,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 alteration_name: str,
                 threshold: float,
                 times: List[float]=None):
        """
        Constructs all the necessary attributes for the RobustnessResult
        object.

        Parameters
        ----------
            steps : List[float]
                list of all the alteration levels that have been applied
                to the input data
            accuracies : List[float]
                list of all the accuracies gathered during the robustness
                analysis
            robustness : float
                robustness of the model over a certain alteration
            title : str
                title of the graph produced by robustness analysis
            xlabel : str
                title of the x axis in the graph produced by robustness
                analysis
            ylabel : str
                title of the y axis in the graph produced by robustness
                analysis
            alteration_name : str
                name of the applied alteration
            threshold : float
                threshold used for robustness computation
            times : List[float]
                list of all the computational times requires for each step
        """
        self.steps = steps
        self.accuracies = accuracies
        self.robustness = robustness
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.alteration_name = alteration_name
        self.threshold = threshold
        self.times = times
