# roby: neural network ROBusteness analYzer

roby (ROBustness analYzer) is a Python tool to perform robustness evaluation of neural network classifiers. Given a trained model, a classified dataset and the list of classes, alterations are applied and the robustness is computed based on the accuracy threshold defined.

## How to use roby

### Requirements

roby has been designed to be more flexible as possible. However, there are some requirements needed for the use of the tool:

1. Alterations must be expressible as input modification between a _minumum_ and a _maximum_ threshold.

2. The model to be tested must be in the _Keras_ format.

3. Only ANNs used for _classification_ are supported by roby.

4. Input data must be representable using `np.ndarray`.

### Installation

roby can be installed using the pip package-management system with
```python
pip install roby
```

## Tutorials and application scenarios

### Tutorial 1: Images classifier - Local execution

In this Tutorial 1, the robustness of a CNN images classifier is analyzed. The execution of all the code is performed locally.

* Create a python file

* Import roby
  ```python
  from roby import *
  ```

* Load your **model** and locate your **images**. The model can be stored in a `.model` or in a `.h5` file and must be in the _Keras_ format.

  ```python
  model = load_model('modelFile.model')
  ```
  Get the images in the test data-set. These can be loaded as a list of paths
  ```python
  file_list = sorted(list(paths.list_images('images')))
  ```
  or as a list of `np.ndarray` as commonly happens when working with already available datasets (eg., in pickle files).

  For each input image we must give the correct label, in order to make possible the evaluation of the results of the CNN when alterations are applied. This can be done in two different ways:
    * Defining a `labeler` function, assigning to each image the correct label
    * Creating a list containing the corresponding label for each input image.

  In this tutorial the first option is used, but more details are reported in the [How to extend roby](#how-to-extend-roby) section of this documentation. Thus, define a labeler function, eg. extracting the label from the file name:
  ```python
  def labeler(image):
    real_label = (image.split('.')[0]).split('_')[-1]
    return real_label
  ```

* Set the **classes** available for classification. This can be done by reading the classes from a `csv` file
  ```python
  classes = set_classes('Classes.csv')
  ```
  or by defining manually a list of strings
  ```python
  classes = ["0", "1", ...]
  ```

* [OPTIONAL] Define your **pre-processing** and/or **post-processing** function. More details on these two functions are reported in the [How to extend roby](#how-to-extend-roby) section of this documentation.

* Define your **environment**
  ```python
  environment = EnvironmentRTest.EnvironmentRTest(model, file_list, classes, labeler_f=labeler)
  ```

* [OPTIONAL] Check the **accuracy** of your model. This value can be used as a baseline of the nominal behavior of your model when no alteration is applied.
  ```python
  accuracy = classification(environment)
  ```

  Note that, if you have defined your dataset as a list of strings, you must specify for the `classification` function a second parameter representing the function specifying how to open the input data file and convert into a `np.ndarray`. For this tutorial, we define a function
  ```python
  def reader(file_name):
    return cv2.imread(file_name)
  ```
  and we check the accuracy with
  ```python
  accuracy = classification(environment, reader)
  ```

* Define the **alteration** against which you want to compute the robustness of your model. For example, if we want to use a _Gaussian Noise_, with variance 200,  we can define:
  ```python
  alteration_type: Alteration = GaussianNoise(0, 1, 200)
  ```

* Set the accuracy **threshold** to be used to compute robustness
  ```python
  accuracy_threshold = 0.8
  ```

* Compute the **robustness** of your model, using 20 points between the minimum and maximum value of the alteration, and get the results.
  ```python
  results = robustness_test(environment, alteration_type, 20,
                                accuracy_threshold)
  display_robustness_results(results)
  ```

  ![Accuracy variation when Gaussian Noise is applied](https://github.com/fmselab/roby/blob/main/docs/images/robustness.jpg?raw=true "Accuracy variation when Gaussian Noise is applied")

### Tutorial 2: Images classifier - Cloud execution

This tutorial analyzes the same case study described in the previous Tutorial but the execution is performed online on Google Colab. The files used for robustness analysis (model file, dataset, ...) must be stored on Google Drive.

* Create a `.ipynb` file on Google Colab.

* Install all the packages, included **roby**, necessary for robustness analysis
  ```python
  !pip install -q keras
  !pip install -U -q PyDrive
  !pip install -q --no-dependencies roby
  ```

* Remove all the **temporary files** from the old computations
  ```python
  !rm -r /content/*
  ```

* Import all the required **libraries**
  ```python
  from google.colab import files
  from roby.CloudTools import authenticate
  from roby.CloudTools import upload_config
  from roby.CloudTools import process_config
  from roby.RobustnessNN import classification
  from roby.RobustnessNN import robustness_test
  from roby.RobustnessNN import display_robustness_results
  from roby.Alterations import GaussianNoise
  from imutils import paths
  import imutils
  import cv2
  from keras.preprocessing.image import img_to_array
  import numpy as np
  ```

* Authenticate with **Google Drive**. We need to authorize Google Colab to access to a Google Drive storage. The user will be required to:
  * Click a link
  * Authorize the connection
  * Paste in a text-box the given Key

  ```python
  drive = authenticate()
  ```

* [OPTIONAL] Define your **pre-processing** and/or **post-processing** function. More details on these two functions are reported in the [How to extend roby](#how-to-extend-roby) section of this documentation.

* Since we work with the same dataset of the previous tutorial, we still need a **labeler** function. Define a labeler function, eg. extracting the label from the file name:
  ```python
  def labeler(image):
    real_label = (image.split('.')[0]).split('_')[-1]
    return real_label
  ```

* Set up your **environment**.
  The `CouldTools` module offers functionalities to load from Google Drive the model, the dataset and, possibly, the csv file containing the classes. To use this functionality, the user has only to specify the three urls:
  ```python
  model_link = "..."
  link_dataset_dir = "..."
  classes_link = "..."
  ```  

  At this point, create the `EnvironmentRTest`:
  ```python
  environment = process_config(model_link, link_dataset_dir, classes_link, drive, labeler_f = labeler)
  ```

* Define the **alteration** against which you want to compute the robustness of your model. For example, if we want to use a _Gaussian Noise_, with variance 200,  we can define:
  ```python
  alteration_type: Alteration = GaussianNoise(0, 1, 200)
  ```

* Set the accuracy **threshold** to be used to compute robustness
  ```python
  accuracy_threshold = 0.8
  ```

* Compute the **robustness** of your model, using 20 points between the minimum and maximum value of the alteration, and get the results.
  ```python
  results = robustness_test(environment, alteration_type, 20,
                                accuracy_threshold)
  display_robustness_results(results)
  ```

### Tutorial 3: Defining customized alterations

If users want to classify data that are not images, or if they want to introduce new kinds of alterations, it is possible to define a customized one. This tutorial will analyze the definition of a new alteration, for audio files, that adds _Gaussian Audio Noise_ to an audio file.

* Extend the `Alterations.Alteration` class and define your new **alteration class**. In this case, since we want to add _Gaussian Noise_ we will need to add a new parameter to the alteration, representing its variance
  ```python
  class AudioNoise(Alteration):   
    def __init__(self, value_from: float, value_to: float, variance: float):
      super().__init__(value_from, value_to)
        self.variance = variance
  ```

* Define the `name()` method in the new alteration class, returning the **name** of the alteration
  ```python
    def name(self) -> str:
      return "AudioNoise"
  ```

* Define the `apply_alteration_data(...)` method in the new alteration class, which applies the defined alteration to an input in the form of `np.ndarray`
  ```python
    def apply_alteration_data(self, data: np.ndarray, alteration_level: float) -> np.ndarray:
      # Apply the alteration
      if float(alteration_level) > 0.0:
        noise=np.random.normal(0, (self.variance/100)*alteration_level, [data.shape[0], data.shape[1], 1])
        data = data + noise
      return data
  ```

* Define the `apply_alteration(...)` method in the new alteration class, which applies the defined alteration to an input when its path is given
  ```python
  def apply_alteration(self, file_name: str, alteration_level: float) -> np.ndarray:
      # Open the file
      file = ...
      return apply_alteration_data(file, alteration_level)
  ```

### Tutorial 4: Sounds classifier




## How to extend roby

* **Loading**:  users can create a testing environment either by giving the path of all the input data
```python
path_list : List[str] = [...]
env = EnvironmentRTest.EnvironmentRTest(model, input_dataset, classes,
                                   label_list=label_list)
```
or by a list of data already in the `np.ndarray` format
```python
path_list : List[np.ndarray] = [...]
env = EnvironmentRTest.EnvironmentRTest(model, input_dataset, classes,
                                  label_list=label_list)
```
If paths are given, for classification purposes, the User shall specify the way to be used to convert the data in the `np.ndarray` format by declaring a
`reader` function:
```python
def reader(file_name: str)
  ...
  return data: np.ndarray
```

* **Labeling**: real labels for input data can be given either with a list of all the labels or by giving a `labeler` function. In the former case, the list `label_list` must be of the same size as the input dataset
```python
env = EnvironmentRTest.EnvironmentRTest(model, input_dataset, classes,
                                   label_list=label_list)
```
while in the latter case the user must define a function receiving a data (in `np.ndarray` format) and returning a string representing the real label
```python
def labeler(image: np.ndarray):
    ...
    return real_label: str
```
and pass this function when defining the environment
```python
env = EnvironmentRTest.EnvironmentRTest(model, file_list, classes,
                                   labeler_f=labeler)
```

* **Alterations**: roby includes the abstract class
```python
Alterations.Alteration
```
that can be extended to create customized alterations. When extending the abstract class, the user must implement the following functions:
  * `name()` returning the name of the alteration.
  * `apply_alteration_data(data, alteration_level)` receiving the input data in the format of `np.ndarray` and returning the data of the same format with the alteration applied
  * `apply_alteration(file_name, alteration_level)` receiving the path of the input data and returning the data with the applied alteration.

* **Pre/Post-processing**: Users can adapt test input data to the ones used for NN training. During the declaration of the test environment users can specify a pre-processing and/or a post-processing function. The former must follow the pattern
```python
def pre_processing(image: np.ndarray):
    ...
    return image: np.ndarray
```
and it is applied to each input data before its recognition by the NN. The latter allows the user to scale the probabilities given as output by the NN. It must follow this pattern
```python
def post_processing(probabilities: List[float]):
    ...
    return new_probabilities:  List[float]
```
The declaration of these two functions is not mandatory. If the user has defined one of them, they can be specified in the declaration of the test environment
```python
env = EnvironmentRTest.EnvironmentRTest(model, file_list, classes,
                                   preprocess_f=pre_processing,
                                   postprocess_f=post_processing,
                                   labeler_f=labeler)
```

### APIs documentation

A full documentation of the roby's APIs can be found at
<https://fmselab.github.io/roby/apis/>

## Examples of use

* Breast cancer images dataset
  <https://github.com/fmselab/roby/tree/main/code/roby_Use/breast-cancer>

* MNIST dataset
  <https://github.com/fmselab/roby/tree/main/code/roby_Use/MNIST>

* German Traffic Signs dataset
  <https://github.com/fmselab/roby/tree/main/code/roby_Use/trafficsigns-imagefailures>

* Numbers recognition from speech
  <https://github.com/fmselab/roby/tree/main/code/roby_Use/speech_recognition>
