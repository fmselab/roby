import setuptools   # type: ignore

setuptools.setup(
    name="roby",
    version="0.0.97",
    author="University of Bergamo",
    author_email="andrea.bombarda@unibg.it",
    description="A general framework to analyse the robustness of a " +
                "Neural Network",
    packages=setuptools.find_packages(),
    install_requires=[
          'opencv_python==4.4.0.46',
          'Pillow==8.0.1',
          'Keras==2.2.4',
          'tensorflow==1.5',
          'sklearn',
          'PyDrive==1.3.1',
          'google-colab',
          'scipy==1.5.4',
          'numpy==1.19.4',
          'oauth2client==4.1.3',
          'matplotlib==3.3.3',
          'protobuf==3.14.0',
          'typing'
    ],)
