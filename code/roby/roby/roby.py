'''
Created on 20 gen 2021
Modified on 21 aug 2024

@author: garganti, bombarda
'''
import click
from keras.models import load_model   # type: ignore
import importlib
from Alterations import Alteration
import EnvironmentRTest
from imutils import paths   # type: ignore
import imutils
import cv2   # type: ignore
from RobustnessNN import robustness_test, set_classes,\
    display_robustness_results, classification

@click.command()
# We support using a different module name and path for custom alterations
@click.option('--module_name', required=False, type=str,  help='module name')
@click.option('--module_path', required=False, type=click.Path(exists=True))
@click.option('--altname', required=True, type=str)
# From to and other parameters for the alteration, like "0,1,100"
@click.option('--altparams', required=True, type=str, help=' parameters for the alteration')
@click.option('--npoints', required=True, type=int)
@click.option('--modelpath', required=True, type=str)
@click.option('--inputpath', required=True, type=str)
@click.option('--theta', required=True, type=float)
@click.option('--labelfile', required=True, type=float)

def reader(file_name):
    return cv2.imread(file_name)

def labeler(image):
    """
    Extracts the label from the image file name
    """
    real_label = (image.split('.')[0]).split('_')[-1]
    return real_label

def runroby(module_name, module_path, altname, altparams, npoints, modelpath, inputpath, theta, labelfile):
    # Load the right module and class for the chosen alteration
    if module_name == None:
        module_name = "Alterations"
        module_path = "Alterations.py"
    module = importlib.import_module(module_name, module_path)

    # Build the alteration of interest
    alterType = getattr(module, altname)
    params = altparams.split(",")
    try:
        alteration = alterType(params)
    except:
        raise Exception("Wrong number of parameters for the alteration")
    
    # Load the model
    model = load_model(modelpath)
    
    # Load the threshold
    accuracy_treshold = theta

    # Load inputs and classes
    file_list = sorted(list(paths.list_images(inputpath)))
    classes = set_classes(labelfile)
    
    # Build the new environment
    environment = EnvironmentRTest(model, file_list, classes,
                                   labeler_f=labeler,
                                   reader_f=reader)
    
    # Get the standard behavior of the net
    accuracy = classification(environment)

    # Perform robustness analysis, with npoints points
    results = robustness_test(environment, alterType, npoints,
                              accuracy_treshold)
    display_robustness_results(results)
 
    
if __name__ == '__main__':
    runroby()
