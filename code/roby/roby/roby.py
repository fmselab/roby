'''
Created on 20 gen 2021

@author: garganti
'''
import click
import importlib
from build.lib.roby.Alterations import Alteration
from build.lib.roby import EnvironmentRTest

@click.command()
@click.option('--module-name', required=False, type=str,  help='module name')
@click.option('--module-path', required=False, type=click.Path(exists=True))
@click.option('--class-name', required=True, type=str)
# from to and other parameters for the alteration, like "0,1,100"
@click.option('--alt-params', required=True, type=str, help=' parameters for the alteration')
@click.option('--step', required=False, type=int)
@click.option('--modelpath', required=False, type=int)
#@click.option('--imagepath', required=False, type=int)

def runroby(module_name, module_path, class_name, alt_params, step, modelpath):
    if module_name == None:
        module_name = "Alterations"
        module_path = "Alterations.py"
    # load the right module and class 
    module = importlib.import_module(module_name, module_path)
    alterType = getattr(module, class_name)
    params = alt_params.split(",")
    # TODO in case there are more parameters
    alteration = alterType(params[0],params[1])
    # build the env
    environment = EnvironmentRTest(modelpath, "")
    # call the method to compute the robustness
    robustness = robustness_test()
 
    
if __name__ == '__main__':
    runroby()
