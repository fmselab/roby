r"""
The **EnvironmentRTest** module contains all the information useful to compute
the robustness analysis of a Neural Network.

In details, the following data are required:

- _model_, i.e. the model on which the robustness analysis will be performed
- _file\_list_, i.e. the list of the images contained into the test dataset.
  In particular this images can be given as a list of paths, or as list of
  np.array images
- _classes_, i.e. the list of the classes in which you want to classify the
  input images
- _label\_list_, i.e. the list of the labels. Its `size` must be equals to the
  `size` of the _file\_list. If you want to use a function to automatically
  assign the label for each input data, you can skip this
  parameter and use the parameter `labeler\_f`.
- _labeler\_f_, i.e. the function used to assign automatically the labels to
  the images. If you already have your own list of labels, you can skip this
  parameter.
- _preprocess\_f_, i.e. a function that can be applied to each picture before
  its classification. You may want to classify raw data, so you can skip this
  parameter.

@author: Andrea Bombarda
"""
from keras import Model   # type: ignore
from typing import Callable, List, TypeVar, Any
import numpy as np   # type: ignore


ImageFile = TypeVar('ImageFile', np.ndarray, str)


class EnvironmentRTest:
    """
    Class with all the elements necessary to execute the robustness analysis
    on our dataset
    """

    def __init__(self, model: Model, file_list: list, classes: List[str],
                 label_list: List[str]=None,
                 labeler_f: Callable[[Any], str]=None,
                 preprocess_f: Callable[[np.ndarray], np.ndarray]=None,
                 reader_f: Callable[[str], np.ndarray]=None):
        """
        Constructs all the necessary attributes for the RobustnessResult object

        Parameters
        ----------
            model : keras.Model
                model we are going to test
            file_list : List[str] or List[np.ndarray]
                list of all the images - str or np.ndarray - (dataset)
                uploaded in our VM
            classes : List[str]
                list with the names - strings - of all the possible
                classification of our model
            label_list: List[str], optional
                list of all the labels associated to each image - str.
                It can be None
            labeler_f : Callable[[ImageFile], str], optional
                labeler function used to get the correct label from an image.
                The image can be given as a str or np.ndarray
                It can be None.
            preprocess_f : Callable[[np.ndarray], np.ndarray], optional
                pre-processing to be executed on the data before the model
                classification. It can be None
            reader_f : Callable[[str], np.ndarray], optional
                optional function used to read the input data from file in a
                np.ndarray format
        """
        self.model = model
        self.file_list = file_list
        self.total_data = len(file_list)
        self.classes = classes
        self.pre_processing = preprocess_f
        self.labeler_f = labeler_f
        self.label_list = label_list
        self.reader_f = reader_f
        # If the label list is not given, then apply the labeler function
        try:
            if self.label_list is None and self.labeler_f is not None:
                assert self.labeler_f is not None
                self.label_list = list(map(labeler_f,        # type: ignore
                                           self.file_list))  # type: ignore
        except ValueError:
            pass
        assert self.label_list is not None and \
            len(self.file_list) == len(self.label_list)
