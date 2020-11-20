"""
In this **cameraFailures** module we create domain specific alterations,
taken from this
[paper](https://github.com/francescosecci/Python_Image_Failures/blob/master/Documenti/Paper.pdf).
"""
from roby.Alterations import Alteration
import cv2   # type: ignore
from PIL import Image   # type: ignore
import numpy as np   # type: ignore


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

    def apply_alteration_image(self, img, alteration_level):
        """
        Method that applies the rain with a given value to the image

        Parameters
        ----------
            img : np.array
                the image on which the rain should be applied
            alterationLevel : float
                the level of the rain that should be applied. It must be
                contained in the range given by the get_range method

        Returns
        -------
            img : np.array
                the altered image on which the rain has been applied
        """
        assert(isinstance(img, np.ndarray))
        if float(alteration_level) > 0.000001:
            # Load the image to be altered
            if isinstance(img, np.ndarray):
                if (self.picture_mode == 'RGB'):
                    img = Image.fromarray(img, 'RGB')
                elif (self.picture_mode == 'L'):
                    img = Image.fromarray((img[:, :, 0]*255).astype('uint8'),
                                          'L')
                else:
                    raise RuntimeError("pictureMode not supported for " +
                                       "brightness alteration")

            # Load the alteration image
            alteration_img = Image.open(
                "Python_Image_Failures\\rain\\rain1.png")

            # Resize the alteration image to the same size of the original one
            alteration_img = change_image_size(img.size[0], img.size[1],
                                               alteration_img)

            # Make sure images got an alpha channel
            alteration_img = alteration_img.convert("RGBA")
            img = img.convert("RGBA")

            # Blend the two images
            img = Image.blend(img, alteration_img, alteration_level)

            # Convert the resulting image in np.ndarray
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        assert(isinstance(img, np.ndarray))
        return img


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

    def apply_alteration_image(self, img, alteration_level):
        """
        Method that applies the Condensation with a given value to the image

        Parameters
        ----------
            img : np.array
                the image on which the Condensation should be applied
            alterationLevel : float
                the level of the Condensation that should be applied. It must
                be contained in the range given by the get_range method

        Returns
        -------
            img : np.array
                the altered image on which the Condensation has been applied
        """
        assert(isinstance(img, np.ndarray))
        if float(alteration_level) > 0.000001:
            # Load the image to be altered
            if isinstance(img, np.ndarray):
                if (self.picture_mode == 'RGB'):
                    img = Image.fromarray(img, 'RGB')
                elif (self.picture_mode == 'L'):
                    img = Image.fromarray((img[:, :, 0]*255).astype('uint8'),
                                          'L')
                else:
                    raise RuntimeError("pictureMode not supported for " +
                                       "brightness alteration")

            # Load the alteration image
            alteration_img = Image.open(
                "Python_Image_Failures\\condensation\\condensation1.png")

            # Resize the alteration image to the same size of the original one
            alteration_img = change_image_size(img.size[0], img.size[1],
                                               alteration_img)

            # Make sure images got an alpha channel
            alteration_img = alteration_img.convert("RGBA")
            img = img.convert("RGBA")

            # Blend the two images
            img = Image.blend(img, alteration_img, alteration_level)

            # Convert the resulting image in np.ndarray
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        assert(isinstance(img, np.ndarray))
        return img
