"""
The **Alteration** module defines the general framework that can be used to
define alterations and provides:

- The basic class **Alteration** that can be extended to create customized
    domain-specific alterations.
- A set of default alterations.

If you want to create your own domain-specific alteration, you have just
to extend the class _Alteration_ and implement the methods `name(self)` and
`applyAlterationImage(self, img, alteration_level)` to comply with your own
alteration.

@author: Andrea Bombarda
"""

import cv2   # type: ignore
from PIL import Image, ImageEnhance, ImageFilter   # type: ignore
import numpy as np   # type: ignore
# For abstract classes
from abc import ABC, abstractmethod
from scipy.ndimage.interpolation import zoom   # type: ignore
import sys
import warnings
from typing import List

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class Alteration(ABC):
    """Abstract Class defining the basic structure for a customized
    alteration."""

    def __init__(self, value_from: float, value_to: float):
        """
        Constructs all the necessary attributes for the Alteration object.

        Parameters
        ----------
            value_from : float
                the minimum alteration value that can be applied
            value_to : float
                the maximum alteration value that can be applied
        """
        self.value_from = value_from
        self.value_to = value_to

    @abstractmethod
    def name(self) -> str:
        """
        Abstract method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """

    def get_range(self, n_values: int) -> np.ndarray:
        """
        Method giving the range of the alteration, starting from 'value_from'
        to 'value_to' with step 'step'

        Parameters
        ----------
            n_values : int
                the values for contained in the range between 'value_from' and
                'value_to', included

        Returns
        -------
            range : np.ndarray
                the range of the alteration, starting from 'value_from' to
                'value_to'
        """
        step = (self.value_to - self.value_from) / n_values
        values = np.arange(self.value_from, self.value_to, step)
        return np.append(values, self.value_to)

    @abstractmethod
    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Abstract method that applies a given alteration with a given value to
        the image

        Parameters
        ----------
            img : np.ndarray
                the image on which the alteration should be applied
            alteration_level : float
                the level of the alteration that should be applied. It must be
                contained in the range given by the getRange method

        Returns
        -------
            img : np.ndarray
                the altered image on which the alteration has been applied
        """

    def apply_alteration(self,
                         file_name: str,
                         alteration_level: float) -> np.ndarray:
        """
        Method that applies a given alteration with a given value to the image,
        whose fileName is given as a parameter

        Parameters
        ----------
            file_name : str
                the path of the image on which the alteration should be applied
            alteration_level : float
                the level of the alteration that should be applied. It must be
                contained in the range given by the get_range method

        Returns
        -------
            image : np.ndarray
                the altered image on which the alteration has been applied
        """
        img = cv2.imread(file_name)
        assert(isinstance(img, np.ndarray))
        return self.apply_alteration_image(img, alteration_level)


class VerticalTranslation(Alteration):
    """
    Class defining the Vertical Translation alteration

    In our experiments we have used (-1, 1) as usual range
    """

    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "VerticalTranslation"

    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the vertical translation with a given value to the
        image
        The image is transformed in a (200x200) image to use a standard format
        when the alteration is applied

        Parameters
        ----------
            img : np.ndarray
                the image on which the vertical translation should be applied
            alteration_level : float
                the level of the vertical translation that should be applied.
                It must be contained in the range given by the get_range method

        Returns
        -------
            img : np.ndarray
                the altered image on which the vertical translation has been
                applied
        """
        assert(isinstance(img, np.ndarray))
        # Zoom the image to be cropped, and then crop it
        if not(-0.000001 <= float(alteration_level) <= 0.000001):
            old_rows, old_cols = img.shape[:-1]
            img = cv2.resize(img, (200, 200))
            img = img[:, 20:-20]
            img = img[20 + int(alteration_level * 20):179 +
                      int(alteration_level * 20), :]
            img = cv2.resize(img, (old_rows, old_cols))

        assert(isinstance(img, np.ndarray))
        return img


class HorizontalTranslation(Alteration):
    """
    Class defining the Horizontal Translation alteration

    In our experiments we have used (-1, 1) as usual range
    """

    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "HorizontalTranslation"

    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the horizontal translation with a given value to
        the image.
        The image is transformed in a (200x200) image to use a standard
        format when the alteration is applied

        Parameters
        ----------
            img : np.ndarray
                the image on which the horizontal translation should be applied
            alteration_level : float
                the level of the horizontal translation that should be applied.
                It must be contained in the range given by the get_range method

        Returns
        -------
            img : np.ndarray
                the altered image on which the horizontal translation has been
                applied
        """
        assert(isinstance(img, np.ndarray))
        if not(-0.000001 <= float(alteration_level) <= 0.000001):
            old_rows, old_cols = img.shape[:-1]
            # Zoom the image to be cropped, and then crop it
            img = cv2.resize(img, (200, 200))
            img = img[20:-20, ]
            img = img[:, 20 + int(alteration_level * 20):179 +
                      int(alteration_level * 20)]
            img = cv2.resize(img, (old_rows, old_cols))

        assert(isinstance(img, np.ndarray))
        return img


class Compression(Alteration):
    """
    Class defining the Compression alteration

    In our experiments we have used (0, 1) as usual range
    """

    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Compression"

    def apply_alteration_image(self,
                               img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the jpeg compression with a given value to the
        image

        Parameters
        ----------
            img : np.ndarray
                the image on which the jpeg compression should be applied
            alteration_level : float
                the level of the jpeg compression that should be applied.
                It must be contained in the range given by the get_range
                method

        Returns
        -------
            img : np.ndarray
                the altered image on which the jpeg compression has been
                applied
        """
        assert(isinstance(img, np.ndarray))
        if alteration_level != 0:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 -
                            (alteration_level * 100)]
            img = cv2.imencode('.jpg', img, encode_param)[1]
            img = cv2.imdecode(img, 1)

        assert(isinstance(img, np.ndarray))
        return img


class GaussianNoise(Alteration):
    """
    Class defining the Gaussian Noise alteration. Tha Gaussian Noise is defined
    as a zero-mean addition with variance 200*alteration_level

    In our experiments we have used (0, 1, 0.025) as usual range
    """

    def __init__(self, value_from: float, value_to: float, variance: float):
        """
        Constructs all the necessary attributes for the Gaussian Noise object.

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
        return "GaussianNoise"

    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the Gaussian Noise with a given value to the image

        Parameters
        ----------
            img : np.ndarray
                the image on which the Gaussian Noise should be applied
            alteration_level : float
                the level of the Gaussian Noise that should be applied. It must
                be contained in the range given
                by the get_range method

        Returns
        -------
            img : np.ndarray
                the altered image on which the Gaussian Noise has been applied
        """
        assert(isinstance(img, np.ndarray))
        if not(-0.000001 <= float(alteration_level) <= 0.000001):
            row, col, ch = img.shape
            mean = 0
            var = self.variance * alteration_level
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            img = img + gauss

        assert(isinstance(img, np.ndarray))
        return img


class Blur(Alteration):
    """
    Class defining the blur alteration.
    The Blur is defined with radius 2 * alteration_level

    In our experiments we have used (0, 1, 0.025) as usual range
    """

    def __init__(self, value_from: float, value_to: float, radius: float=2,
                 picture_mode: str='RGB'):
        """
        Constructs all the necessary attributes for the Gaussian Noise object.

        Parameters
        ----------
            value_from : float
                the minimum alteration value that can be applied
            value_to : float
                the maximum alteration value that can be applied
            radius : float, optional
                the radius to be used to apply the blur. Equals to 2 by default
            picture_mode : str, optional
                the picture mode used to represent the images in the
                np.ndarray.
                It is 'RGB' by default. Set this value to 'L' if your image is
                represented using a np.ndarray with values float32
                and scaled within 0 and 1
        """
        super().__init__(value_from, value_to)
        self.radius = radius
        self.picture_mode = picture_mode

    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Blur"

    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the Blur with a given value to the image

        Parameters
        ----------
            img : np.ndarray
                the image on which the Blur should be applied
            alteration_level : float
                the level of the Blur that should be applied. It must be
                contained in the range given by the
                get_range method

        Returns
        -------
            img : np.ndarray
                the altered image on which the Blur has been applied
        """
        assert(isinstance(img, np.ndarray))
        if alteration_level != 0:
            if isinstance(img, np.ndarray):
                if self.picture_mode == 'RGB':
                    img = Image.fromarray(img, 'RGB')
                elif self.picture_mode == 'L':
                    img = Image.fromarray((img[:, :, 0]*255).astype('uint8'),
                                          'L')
                else:
                    raise RuntimeError("pictureMode not supported for blur " +
                                       "alteration")

            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius *
                                                      alteration_level))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        assert(isinstance(img, np.ndarray))
        return img

    def apply_alteration(self,
                         file_name: str,
                         alteration_level: float) -> np.ndarray:
        """
        Method that applies a blur alteration with a given value to the image,
        whose fileName is given as a parameter

        Parameters
        ----------
            file_name : str
                the path of the image on which the blur alteration should be
                applied
            alteration_level : float
                the level of the blur alteration that should be applied.
                It must be contained in the range given by
                the get_range method

        Returns
        -------
            image : np.ndarray
                the altered image on which the blur alteration has been applied
        """
        if alteration_level != 0:
            img = Image.open(file_name)
            img = np.asarray(img)
        else:
            img = cv2.imread(file_name)
        assert(isinstance(img, np.ndarray))
        return self.apply_alteration_image(img, alteration_level)


class Brightness(Alteration):
    """
    Class defining the Brightness Variation alteration.
    It is defined as a brightness enhancement of 0.5*alteration_level

    In our experiments we have used (-1, 1, 0.05) as usual range
    """

    def __init__(self, value_from: float, value_to: float,
                 picture_mode: str='RGB'):
        """
        Constructs all the necessary attributes for the Gaussian Noise object.

        Parameters
        ----------
            value_from : float
                the minimum alteration value that can be applied
            value_to : float
                the maximum alteration value that can be applied
            picture_mode : str, optional
                the picture mode used to represent the images in the
                np.ndarray.
                It is 'RGB' by default.
                Set this value to 'L' if your image is represented using a
                np.ndarray with values float32 and scaled within 0 and 1
        """
        super().__init__(value_from, value_to)
        self.picture_mode = picture_mode

    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Brightness"

    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the Brightness Variation with a given value to the
        image

        Parameters
        ----------
            img : np.ndarray
                the image on which the Brightness Variation should be applied
            alteration_level : float
                the level of the Brightness Variation that should be applied.
                It must be contained in the range
                given by the get_range method

        Returns
        -------
            img : np.ndarray
                the altered image on which the Brightness Variation has been
                applied
        """
        assert(isinstance(img, np.ndarray))
        if not (-0.00001 <= float(alteration_level) <= 0.00001):
            if isinstance(img, np.ndarray):
                if self.picture_mode == 'RGB':
                    img = Image.fromarray(img, 'RGB')
                elif self.picture_mode == 'L':
                    img = Image.fromarray((img[:, :, 0]*255).astype('uint8'),
                                          'L')
                else:
                    raise RuntimeError("picture_mode not supported for " +
                                       "brightness alteration")
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1 + (alteration_level * 0.5))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        assert(isinstance(img, np.ndarray))
        return img

    def apply_alteration(self,
                         file_name: str,
                         alteration_level: float) -> np.ndarray:
        """
        Method that applies a Brightness Variation with a given value to the
        image, whose fileName is given as a parameter

        Parameters
        ----------
            file_name : str
                the path of the image on which the Brightness Variation should
                be applied
            alteration_level : float
                the level of the Brightness Variation that should be applied.
                It must be contained in the range given by the get_range method

        Returns
        -------
            image : np.ndarray
                the altered image on which the Brightness Variation has been
                applied
        """
        if alteration_level != 0:
            img = Image.open(file_name)
            img = np.asarray(img)
        else:
            img = cv2.imread(file_name)
        assert(isinstance(img, np.ndarray))
        return self.apply_alteration_image(img, alteration_level)


class Zoom(Alteration):
    """
    Class defining the Zoom alteration.
    The image is zoomed and, after that, cut to obtain an image with the same
    dimensions than the initial one.

    In our experiments we have used (0, 1, 0.025) as usual range
    """

    def name(self) -> str:
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Zoom"

    def apply_alteration_image(self, img: np.ndarray,
                               alteration_level: float) -> np.ndarray:
        """
        Method that applies the Zoom with a given value to the image

        Parameters
        ----------
            img : np.ndarray
                the image on which the Zoom should be applied
            alteration_level : float
                the level of the Zoom that should be applied. It must be
                contained in the range given by the get_range method

        Returns
        -------
            img : np.ndarray
                the altered image on which the Zoom has been applied
        """
        assert(isinstance(img, np.ndarray))
        if -0.000001 <= float(alteration_level) <= 0.000001:
            return img

        # For multi-channel images we don't want to apply the zoom factor to
        # the RGB dimension, so instead we create a tuple of zoom factors,
        # one per array dimension, with 1's for any trailing dimensions after
        # the width and height.
        h, w = img.shape[:2]
        zoom_tuple = (float(1 + alteration_level),) * 2 + (1,) * (img.ndim - 2)
        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / float(1 + alteration_level)))
        zw = int(np.round(w / float(1 + alteration_level)))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        img = out

        assert(isinstance(img, np.ndarray))
        return img


class AlterationSequence:
    """
    Class defining the a sequence of alterations, each one with its own
    alteration_level
    """

    def __init__(self, alterations: List[Alteration],
                 alteration_levels: List[float]):
        """
        Constructs all the necessary attributes for the AlterationSequence
        object.

        Parameters
        ----------
            alterations : List[Alteration]
                list of alterations to be applied
            alteration_levels : List[float]
                list of alteration levels
        Raises
        ------
            ValueError
                when alterations and alteration_levels have different sizes
        """
        if len(alterations) != len(alteration_levels):
            raise ValueError("alterations and alteration_levels must be of " +
                             "the same size")
        self.alterations = alterations
        self.alteration_levels = alteration_levels

    def name(self) -> str:
        """
        Method to get the alteration sequence name

        Returns
        -------
            alteration_name : str
                the name of the alteration type
        """
        alteration_name = "Seq:"
        for a in self.alterations:
            alteration_name = alteration_name + " " + a.name() + ","
        return alteration_name

    def apply_alteration(self, img: np.ndarray) -> np.ndarray:
        """
        Method that applies a given sequence of alterations with given values
        to the image

        Parameters
        ----------
            img : np.ndarray
                the image on which the alterations should be applied

        Returns
        -------
            img : np.ndarray
                the altered image on which the alterations have been applied
        """
        alteration_counter = 0
        for a in self.alterations:
            img = a.apply_alteration_image(img, self.alteration_levels[
                alteration_counter])
            alteration_counter = alteration_counter + 1
        return img
