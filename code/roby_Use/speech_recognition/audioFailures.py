from roby.Alterations import Alteration
import cv2   # type: ignore
import numpy as np   # type: ignore
from cmath import sqrt
from numpy import shape, reshape

class AudioNoise(Alteration):   
    
    def __init__(self, value_from: float, value_to: float, variance: float):
        """
        Constructs all the necessary attributes for the Audio Noise object.

        Parameters
        ----------
            value_from : float
                the minimum alteration value that can be applied
            value_to : float
                the maximum alteration value that can be applied
            variance : float
                the variance to be used in the Gaussian Noise generation
        """
        super().__init__(value_from, value_to)
        self.variance = variance
    
    
    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "AudioNoise"
    
    
    def apply_alteration(self, data, alteration_level):
        """
        Method that applies the rain with a given value to the image

        Parameters
        ----------
            data : np.array
                the data on which the noise should be applied
            alterationLevel : float
                the level of the noise that should be applied. It must be
                contained in the range given by the get_range method

        Returns
        -------
            data : np.array
                the altered data on which the noise has been applied
        """
        assert(isinstance(data, np.ndarray))
        if float(alteration_level) > 0.000001:
            noise=np.random.normal(0, (self.variance/100)*alteration_level, [data.shape[0], data.shape[1], 1])
            data = data + noise 

        assert(isinstance(data, np.ndarray))
        return data
