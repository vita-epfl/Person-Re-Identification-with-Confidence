from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *

from PIL import Image
import random
import numpy as np

from torchvision.transforms import functional as F

class RandomSizedEarser(object):
    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.5):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(-1, 1.0)
        W = img.size[0]
        H = img.size[1]
        area = H * W

        if p1 > self.p:
            return img
        else:
            gen = True
            while gen:
                Se = random.uniform(self.sl, self.sh)*area
                re = random.uniform(self.asratio, 1/self.asratio)
                He = np.sqrt(Se*re)
                We = np.sqrt(Se/re)
                xe = random.uniform(0, W-We)
                ye = random.uniform(0, H-He)
                if xe+We <= W and ye+He <= H and xe>0 and ye>0:
                    x1 = int(np.ceil(xe))
                    y1 = int(np.ceil(ye))
                    x2 = int(np.floor(x1+We))
                    y2 = int(np.floor(y1+He))
                    part1 = img.crop((x1, y1, x2, y2))
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    I = Image.new('RGB', part1.size, (Rc, Gc, Bc))
                    img.paste(I, part1.size)
                    return img

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class RandomHorizontalFlip_custom(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), True
        return img, False

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
