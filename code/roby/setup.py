import setuptools   # type: ignore

setuptools.setup(
    name="roby",
    version="0.0.4",
    author="University of Bergamo",
    author_email="andrea.bombarda@unibg.it",
    description="A general framework to analyse the robustness of a " +
                "Neural Network",
    packages=setuptools.find_packages(),
    python_requires='>=3.6, <=3.9.11',
    install_requires=[
        'numpy==1.26.4',
        'opencv_python>=4.0',
        'Pillow>=5.4.1',
        'Keras>=2.2.4',
        'tensorflow==2.15.1',
        'scikit-learn',
        'PyDrive>=1.3.1',
        'scipy>=1.4.1',
        'oauth2client>=4.1.3',
        'matplotlib>=3.0.2',
        'protobuf==3.20.3',
        'typing>=3.7.4',
        'imutils>=0.5.2',
        'sympy'])
