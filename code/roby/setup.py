import setuptools   # type: ignore

setuptools.setup(
    name="roby",
    version="0.0.8",
    author="University of Bergamo",
    author_email="andrea.bombarda@unibg.it",
    description="A general framework to analyse the robustness of a " +
                "Neural Network",
    packages=setuptools.find_packages(),
	python_requires='>=3.6, <=3.8.6',
    install_requires=[ 
          'numpy==1.18.5',
          'opencv_python>=4.0',
          'Pillow>=5.4.1',
          'Keras>=2.2.4',
          'tensorflow==2.3.1',
          'sklearn',
          'PyDrive>=1.3.1',
          'scipy>=1.4.1',
          'oauth2client>=4.1.3',
          'matplotlib>=3.0.2',
          'protobuf==3.13.0',
          'typing>=3.7.4',
		  'imutils>=0.5.2'
    ],)
