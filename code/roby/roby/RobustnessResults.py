"""
The **RobustnessResults** module contains the class which is used to store
the results of each robustness analysis.
"""


class RobustnessResults:
    """Class storing the results of the robustness."""

    def __init__(self, steps: list, accuracies: list, robustness: float,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 alteration_name: str):
        """
        Constructs all the necessary attributes for the RobustnessResult
        object.

        Parameters
        ----------
            steps : list
                list of all the alteration levels that have been applied
                to the image
            accuracies : list
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
        """
        self.steps = steps
        self.accuracies = accuracies
        self.robustness = robustness
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.alteration_name = alteration_name
