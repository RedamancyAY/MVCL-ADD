# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numbers
import random

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms


# -

# * brightness (float or tuple of float (min, max)): How much to jitter brightness.
#     * brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
#     * brightness_factor: 0 gives a black image, **1 gives the original image** while 2 increases the brightness by a factor of 2.

# * contrast (float or tuple of float (min, max)): How much to jitter contrast.
#     * contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
#     * 0 gives a solid gray image, **1 gives the original image** while 2 increases the contrast by a factor of 2.

# * saturation (float or tuple of float (min, max)): How much to jitter saturation.
#     * saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
#     * 0 will give a black and white image, **1 gives the original image**  while 2 will enhance the saturation by a factor of 2.

# * hue (float or tuple of float (min, max)): How much to jitter hue.
#     * hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
#     * Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give an image with complementary colors while **0 gives the original image**.

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image. --modified from pytorch source code
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5
    ):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.threshold = p

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_brightness(img, brightness_factor)
                )
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_contrast(img, contrast_factor)
                )
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_saturation(img, saturation_factor)
                )
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor))
            )

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def color_jitter(self, x):
        B = x.shape[0]
        if x.ndim == 5:
            x = x.permute(0, 2, 1, 3, 4) # (N, C, T, H, W) -> (N, T, C, H, W)
            
        x = torch.stack(
            [
                self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )(x[i])
                for i in range(B)
            ],
            dim=0,
        )
        
        if x.ndim == 5:
            x = x.permute(0, 2, 1, 3, 4)# (N, T, C, H, W) -> (N, C, T, H, W)
            
        return x
    
    def __call__(self, x, label=None):
        assert x.ndim in [4, 5]
        
        B = x.shape[0]
        p = torch.rand(B)
        index1 = torch.where(p <= self.threshold)
        index2 = torch.where(p > self.threshold)
        if len(index1[0]) == 0:
            return x
        else:
            x[index1] = self.color_jitter(x[index1])
            return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string

# + tags=["active-ipynb", "style-student"]
# model = ColorJitter(
#     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
# )
# x = torch.rand(16, 3, 10, 224, 224)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# x = x.to(device)
#
# y = model(x)
