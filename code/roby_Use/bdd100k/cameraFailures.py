"""
In this **cameraFailures** module we create domain specific alterations,
taken from this
[paper](https://github.com/francescosecci/Python_Image_Failures/blob/master/Documenti/Paper.pdf).
"""
from roby.Alterations import Alteration
import cv2   # type: ignore
from PIL import Image   # type: ignore
import numpy as np   # type: ignore
from PIL import Image, ImageEnhance, ImageFilter
import skimage as ski
from scipy.ndimage import gaussian_filter


def change_image_size(max_width, max_height, image):
    """
    Converts the size of an image to fit inside another image. It is use to
    blend two images having the same maximum size.
    """
    widthRatio = max_width / image.size[0]
    heightRatio = max_height / image.size[1]

    new_width = int(widthRatio * image.size[0])
    new_height = int(heightRatio * image.size[1])

    new_image = image.resize((new_width, new_height))
    return new_image


class RainAlteration_1(Alteration):

    def __init__(self, picture_mode):
        """
        Constructs all the necessary attributes for the RainAlteration_1
        object.
        It is based on the blending technique, thus it is applied in the range
        (0,1) where 0 means that the original image is kept, 1 means that the
        alteration image is kept. Numbers between 0 and 1 creates a blending
        between the two images, based on the blending factor

        Parameters
        ----------
            picture_mode : str
                    the picture mode used to represent the images in the
                    np.array. It is 'RGB' by default.
                    Set this value to 'L' if your image is represented using a
                    np.array with values float32 and scaled within 0 and 1
        """
        super().__init__(0, 0.2)
        self.picture_mode = picture_mode

    def name(self):
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Rain_1"

    def apply_alteration(self, data, alteration_level):
        """
        Method that applies the rain with a given value to the image

        Parameters
        ----------
            data : np.array
                the data on which the rain should be applied
            alterationLevel : float
                the level of the rain that should be applied. It must be
                contained in the range given by the get_range method

        Returns
        -------
            data : np.array
                the altered data on which the rain has been applied
        """
        assert(isinstance(data, np.ndarray))
        if float(alteration_level) > 0.000001:
            # Load the image to be altered
            if isinstance(data, np.ndarray):
                if (self.picture_mode == 'RGB'):
                    data = Image.fromarray(data, 'RGB')
                elif (self.picture_mode == 'L'):
                    data = Image.fromarray((data[:, :, 0]*255).astype('uint8'),
                                           'L')
                else:
                    raise RuntimeError("pictureMode not supported for " +
                                       "brightness alteration")

            # Load the alteration image
            alteration_img = Image.open(
                "Python_Image_Failures/rain/rain1.png")

            # Resize the alteration image to the same size of the original one
            alteration_img = change_image_size(data.size[0], data.size[1],
                                               alteration_img)

            # Make sure images got an alpha channel
            alteration_img = alteration_img.convert("RGBA")
            data = data.convert("RGBA")

            # Blend the two images
            data = Image.blend(data, alteration_img, alteration_level)

            # Convert the resulting image in np.ndarray
            data = np.array(data)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        assert(isinstance(data, np.ndarray))
        return data


class Condensation_1(Alteration):

    def __init__(self, picture_mode):
        """
        Constructs all the necessary attributes for the Condensation_1 object.
        It is based on the blending technique, thus it is applied in the range
        (0,1) where 0 means that the original image is kept, 1 means that the
        alteration image is kept. Numbers between 0 and 1 creates a blending
        between the two images, based on the blending factor

        Parameters
        ----------
            picture_mode : str
                    the picture mode used to represent the images in the
                    np.array. It is 'RGB' by default. Set this value to 'L'
                    if your image is represented using a np.array with values
                    float32 and scaled within 0 and 1
        """
        super().__init__(0, 0.2)
        self.picture_mode = picture_mode

    def name(self):
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Condensation_1"

    def apply_alteration(self, data, alteration_level):
        """
        Method that applies the Condensation with a given value to the image

        Parameters
        ----------
            data : np.array
                the data on which the Condensation should be applied
            alterationLevel : float
                the level of the Condensation that should be applied. It must
                be contained in the range given by the get_range method

        Returns
        -------
            data : np.array
                the altered data on which the Condensation has been applied
        """
        assert(isinstance(data, np.ndarray))
        if float(alteration_level) > 0.000001:
            # Load the image to be altered
            if isinstance(data, np.ndarray):
                if (self.picture_mode == 'RGB'):
                    data = Image.fromarray(data, 'RGB')
                elif (self.picture_mode == 'L'):
                    data = Image.fromarray((data[:, :, 0]*255).astype('uint8'),
                                           'L')
                else:
                    raise RuntimeError("pictureMode not supported for " +
                                       "brightness alteration")

            # Load the alteration image
            alteration_img = Image.open(
                "Python_Image_Failures/condensation/condensation1.png")

            # Resize the alteration image to the same size of the original one
            alteration_img = change_image_size(data.size[0], data.size[1],
                                               alteration_img)

            # Make sure images got an alpha channel
            alteration_img = alteration_img.convert("RGBA")
            data = data.convert("RGBA")

            # Blend the two images
            data = Image.blend(data, alteration_img, alteration_level)

            # Convert the resulting image in np.ndarray
            data = np.array(data)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        assert(isinstance(data, np.ndarray))
        return data

class CustomBrightness(Alteration):
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

    def apply_alteration(self, data: np.ndarray,
                         alteration_level: float) -> np.ndarray:
        """
        Method that applies the Brightness Variation with a given value to the
        data

        Parameters
        ----------
            data : np.ndarray
                the data on which the Brightness Variation should be applied
            alteration_level : float
                the level of the Brightness Variation that should be applied.
                It must be contained in the range
                given by the get_range method

        Returns
        -------
            data : np.ndarray
                the altered data on which the Brightness Variation has been
                applied
        """
        assert(isinstance(data, np.ndarray))
        if not (-0.0001 <= float(alteration_level) <= 0.0001):
            if isinstance(data, np.ndarray):
                data = ski.exposure.adjust_gamma(data, gain=alteration_level + 1)
            
            data = np.array(data)

        assert(isinstance(data, np.ndarray))
        return data

class CustomBlur(Alteration):
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

    def apply_alteration(self, data: np.ndarray,
                         alteration_level: float) -> np.ndarray:
        """
        Method that applies the Blur with a given value to the data

        Parameters
        ----------
            data : np.ndarray
                the data on which the Blur should be applied
            alteration_level : float
                the level of the Blur that should be applied. It must be
                contained in the range given by the
                get_range method

        Returns
        -------
            data : np.ndarray
                the altered data on which the Blur has been applied
        """
        assert(isinstance(data, np.ndarray))
        if alteration_level != 0.0:
            if isinstance(data, np.ndarray):
                if self.picture_mode == 'float64':
                    data = gaussian_filter(data, sigma=self.radius *
                                                        alteration_level)
                else:
                    raise RuntimeError("pictureMode not supported for blur " +
                                       "alteration")

            data = np.array(data)

        assert(isinstance(data, np.ndarray))
        return data

class Ice_1(Alteration):

    def __init__(self, picture_mode):
        """
        Constructs all the necessary attributes for the Ice_1 object.
        It is based on the blending technique, thus it is applied in the range
        (0,1) where 0 means that the original image is kept, 1 means that the
        alteration image is kept. Numbers between 0 and 1 creates a blending
        between the two images, based on the blending factor

        Parameters
        ----------
            picture_mode : str
                    the picture mode used to represent the images in the
                    np.array. It is 'RGB' by default. Set this value to 'L'
                    if your image is represented using a np.array with values
                    float32 and scaled within 0 and 1
        """
        super().__init__(0, 0.2)
        self.picture_mode = picture_mode

    def name(self):
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Ice_1"

    def apply_alteration(self, data, alteration_level):
        """
        Method that applies the Ice alteration with a given value to the image

        Parameters
        ----------
            data : np.array
                the data on which the Ice alteration should be applied
            alterationLevel : float
                the level of the Ice alteration that should be applied. It must
                be contained in the range given by the get_range method

        Returns
        -------
            data : np.array
                the altered data on which the Ice alteration has been applied
        """
        assert(isinstance(data, np.ndarray))
        if float(alteration_level) > 0.000001:
            # Load the image to be altered
            if isinstance(data, np.ndarray):
                if (self.picture_mode == 'RGB'):
                    data = Image.fromarray(data, 'RGB')
                elif (self.picture_mode == 'L'):
                    data = Image.fromarray((data[:, :, 0]*255).astype('uint8'),
                                           'L')
                else:
                    raise RuntimeError("pictureMode not supported for " +
                                       "Ice alteration")

            # Load the alteration image
            alteration_img = Image.open(
                "Python_Image_Failures/ice/ice3.png")

            # Resize the alteration image to the same size of the original one
            alteration_img = change_image_size(data.size[0], data.size[1],
                                               alteration_img)

            # Make sure images got an alpha channel
            alteration_img = alteration_img.convert("RGBA")
            data = data.convert("RGBA")

            # Blend the two images
            data = Image.blend(data, alteration_img, alteration_level)

            # Convert the resulting image in np.ndarray
            data = np.array(data)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        assert(isinstance(data, np.ndarray))
        return data