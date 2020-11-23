# roby: neural network ROBusteness analYzer

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


### APIs documentation

<https://fmselab.github.io/roby/apis/>