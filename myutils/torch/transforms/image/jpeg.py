# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + editable=true slideshow={"slide_type": ""}
import torch
import torchvision.transforms as T
from PIL import Image
import io
import random
import numpy as np


# -

class RandomJPEGCompression:
    """Apply random JPEG compression to an input image.

    Args:
        quality_range (tuple): Range of JPEG quality levels to randomly choose from.
                               Lower values mean more compression (lower quality).
    """

    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def apply(self, image):
        return self(image)
        
    
    def __call__(self, image):
        """Apply the JPEG compression.

        Args:
            img (PIL Image, torch.Tensor, or numpy.ndarray): Input image to transform.

        Returns:
            PIL Image, torch.Tensor, or numpy.ndarray: JPEG compressed image in the same format as input.
        """
        img = image
        input_type = type(img)

        if isinstance(img, np.ndarray):
            # Convert NumPy array to PIL Image
            img = Image.fromarray(img)

        elif isinstance(img, torch.Tensor):
            # Convert tensor to PIL Image
            img = T.ToPILImage()(img)

        # Randomly select a JPEG quality level within the given range
        quality = random.randint(*self.quality_range)

        # print(quality)
        
        # Compress the image using JPEG format with the selected quality level
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)

        # Convert the compressed image back to the original format
        if input_type == np.ndarray:
            compressed_img = np.array(compressed_img)

        elif input_type == torch.Tensor:
            compressed_img = T.ToTensor()(compressed_img)

        return {'image' :compressed_img}

# +
# # Example NumPy array (e.g., an image with shape HxWxC)
# input_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

# # Instantiate the transform
# jpeg_compression = RandomJPEGCompression(quality_range=(50, 90))

# # Apply the transform to the NumPy array
# compressed_image = jpeg_compression(input_image)
# # Now `compressed_image` is a NumPy array with JPEG compression applied
