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
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from empatches import EMPatches


def change_image_size(max_width, max_height, image):
    """
    Converts the size of an image to fit inside another image. It is use to
    blend two images having the same maximum size.
    """
    widthRatio = max_width / image.shape[0]
    heightRatio = max_height / image.shape[1]

    new_width = int(widthRatio * image.shape[0])
    new_height = int(heightRatio * image.shape[1])

    new_image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return new_image


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
        super().__init__(0, 0.5)
        self.picture_mode = picture_mode
        self.alteration_img = Image.open("Python_Image_Failures/condensation/condensation3.png")
        # Load the alteration image into an array
        self.alteration_img = np.array(self.alteration_img, np.uint8)      

        # Extract patches from the alteration image
        emp = EMPatches()
        self.img_patches, indices = emp.extract_patches(self.alteration_img, patchsize=32, overlap=0.4)
        # Remove from img_patches all the patches that have all elements equal to zero
        self.img_patches = [patch for patch in self.img_patches if not np.sum(patch)<10000]

        # Alteration levels for each input
        self.alterations = {}

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
            if not isinstance(data, np.ndarray):
                raise RuntimeError("pictureMode not supported for " +
                    "condensation alteration")

            # Extract one of the patches randomly if not already set previously
            if self.alterations.get(str(data)) is None:
                patch = self.img_patches[np.random.randint(0, len(self.img_patches))]
                self.alterations[str(data)] = patch
            else:
                patch = self.alterations.get(str(data))

            # Normalize image_A to [0, 255] and convert to uint8
            image_A_normalized = np.clip(data * 255, 0, 255).astype(np.uint8)

            # Convert the NumPy array to a PIL Image
            image_A_pil = Image.fromarray(image_A_normalized)

            # Ensure image_B is in RGBA mode
            image_B = Image.fromarray(patch, 'RGBA')

            # Resize image_A_pil to match image_B if necessary
            if image_A_pil.size != image_B.size:
                image_A_pil = image_A_pil.resize(image_B.size, Image.ANTIALIAS)

            # Perform alpha blending
            blended_image = Image.blend(image_A_pil.convert('RGBA'), image_B, alteration_level)

            # Convert the blended image back to a NumPy array
            blended_image_np = np.array(blended_image)
            blended_image_np_rgb = blended_image_np[..., :3]
            blended_image_np_rgb_float64 = blended_image_np_rgb.astype(np.float64) / 255.0

            data = blended_image_np_rgb_float64

        assert(isinstance(data, np.ndarray))
        return data


class RainAlteration_1(Condensation_1):

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
        super().__init__(picture_mode)
        self.picture_mode = picture_mode
        self.alteration_img = Image.open("Python_Image_Failures/rain/rain3.png")
        # Load the alteration image into an array
        self.alteration_img = np.array(self.alteration_img, np.uint8)      

        # Extract patches from the alteration image
        emp = EMPatches()
        self.img_patches, indices = emp.extract_patches(self.alteration_img, patchsize=32, overlap=0.4)
        # Remove from img_patches all the patches that have all elements equal to zero
        self.img_patches = [patch for patch in self.img_patches if not np.sum(patch)<10000]

    def name(self):
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Rain_1"
    

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
        if not (-0.0001 <= float(alteration_level) <= 0.0001):
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

class Ice_1(Condensation_1):

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
        super().__init__(picture_mode)
        self.picture_mode = picture_mode
        self.alteration_img = Image.open("Python_Image_Failures/ice/ice3.png")
        # Load the alteration image into an array
        self.alteration_img = np.array(self.alteration_img, np.uint8)      

        # Extract patches from the alteration image
        emp = EMPatches()
        self.img_patches, indices = emp.extract_patches(self.alteration_img, patchsize=32, overlap=0.4)
        # Remove from img_patches all the patches that have all elements equal to zero
        self.img_patches = [patch for patch in self.img_patches if not np.sum(patch)<10000]

    def name(self):
        """
        Method to get the alteration name

        Returns
        -------
            alterationName : str
                the name of the alteration type
        """
        return "Ice_1"