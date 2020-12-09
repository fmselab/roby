"""
**roby** (ROBustness Analyzer) is a library containing
a general framework that can be used to evaluate the robustness of a NN
w.r.t. alterations.

Some of these alterations are provided by the framework, but the user can
define its own alteration by extending the class _Alteration_ and
implementing the methods `name(self)` and
`apply_alteration(self, data, alteration_level)`.

The followings are the alterations provided by **roby**:

- Blur
- Brightness
- Compression
- Gaussian Noise
- Horizontal Translation
- Vertical Translation
- Zoom

The general procedure to use the **roby** framework is described by
the following steps:

1. Load your model. The model must be a `keras.model` or can be loaded
   from an `.h5` file.
2. Load your test dataset. This can be done using a list of strings
   representing the paths of the input data, or loading directly the
   input data in a np.ndarray list.
3. Load the classes in which you want to classify your input data. This
   can be done either by loading them from a `.csv` file or directly
   in a list.
4. Load the labels of your input data. These can be loaded in a list
   of strings (with the same length of the test dataset list), or by
   giving a customized labeler function.
5. Optionally, define your `pre-processing` function. It is
   used to adapt the input data to the format
   of those on which the network has been previously trained.
6. Define your testing environment.
7. Define the alteration w.r.t. you want to test the robustness of your
   network.
8. Launch the robustness test and show the results.

@author: Andrea Bombarda
"""
