# roby: neural network ROBusteness analYzer

Spiegazione di roby ,...

## How to use roby

### Requirements

roby has been designed to be more flexible as possible. However, there are some requirements needed for the use of the tool:

1. Alterations must be expressible as input modification between a _minumum_ and a _maximum_ threshold.

2. The model to be tested must be in the _Keras_ format.

3. Only ANNs used for _classification_ are supported by roby.

4. Input data must be representable using _np.ndarrays_.

## Tutorials and application scenarios

### tutorial 1: images classifier

### brief tutorial on how to use roby

1. create a python file

2. import roby

3. load your model and locate your images

```python
model = load_model('model\\Medical.model')
```
get the images in the test data-set
```python
file_list = sorted(list(paths.list_images('images')))
```
set the classes
```python
classes = set_classes('model\\Classes.csv')
```

4. define your environment

load the environment
```python
environment = EnvironmentRTest(model, file_list, classes, labeler_f=labeler)
```


5. check the current accuracy of your model

Get the standard behavior of the net
```python
accuracy = classification(environment)
```


6. check the robustness against a desired alteration

create the alteration_type as a GaussianNoise with variance 200
```python
alteration_type: Alteration = GaussianNoise(0, 1, 200)
```

Set the accuracy threshold
```python
accuracy_treshold = 0.8
```

robustness analysis, with 20 points
```python
results = robustness_test(environment, alteration_type, 20,
                              accuracy_treshold)
display_robustness_results(results)
```

![alt text](images/robustenss.jpg "")


### tutorial XX: using colab



## How to extend roby

* **Loading**:

* **Labeling**:

* **Alterations**:

* **Pre/Post-processing**: Users can adapt test input data to the ones used for NN training. During the declaration of the test environment users can specify a pre-processing and/or a post-processing function. The former must follow the following pattern
```python
def pre_processing(image: np.ndarray):
    ...
    return image: np.ndarray
```
and it is applied to each input data before its recognition by the NN. The latter allows the user to scale the probabilities given as output by the NN.

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
