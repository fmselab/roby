import setuptools   # type: ignore

setuptools.setup(
    name="roby",
    version="0.0.5",
    author="University of Bergamo",
    author_email="andrea.bombarda@unibg.it",
    description="A general framework to analyse the robustness of a " +
                "Neural Network",
    packages=setuptools.find_packages(),
	python_requires='>=3.6, <=3.8',
    install_requires=[
          'opencv_python>=4.0',
          'Pillow>=5.4.1',
          'Keras>=2.2.4',
          'tensorflow>=1.5',
          'sklearn',
          'PyDrive>=1.3.1',
          'google-colab>=1.0',
          'scipy>=1.4.1',
          'numpy>=1.16.1',
          'oauth2client>=4.1.3',
          'matplotlib>=3.0.2',
          'protobuf>=3.13.0',
          'typing>=3.7.4',
		  'imutils>=0.5.2'
    ],)
