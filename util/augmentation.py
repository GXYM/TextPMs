import numpy as np
import math
import cv2
import copy
import numpy.random as random

from shapely.geometry import Polygon



####<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<####
####<<<<<<<<<<<  Class  >>>>>>>>>>>>>####
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>####
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons


class ResizeSquare(object):
    def __init__(self, size=(480, 1280)):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        img_size_min = min(h, w)
        img_size_max = max(h, w)

        if img_size_min < self.size[0]:
            im_scale = float(self.size[0]) / float(img_size_min)  # expand min to size[0]
            if np.round(im_scale * img_size_max) > self.size[1]:  # expand max can't > size[1]
                im_scale = float(self.size[1]) / float(img_size_max)
        elif img_size_max > self.size[1]:
            im_scale = float(self.size[1]) / float(img_size_max)
        else:
            im_scale = 1.0

        new_h = int(int(h * im_scale/32)*32)
        new_w = int(int(w * im_scale/32)*32)
        #new_h = int(h * im_scale)
        #new_w = int(w * im_scale)
        if new_h*new_w >= 1600*1600:
            im_scale = 1600/float(img_size_max)
            new_h = int(int(h * im_scale/32)*32)
            new_w = int(int(w * im_scale/32)*32)
            #new_h = int(h * im_scale)
            #new_w = int(w * im_scale)

        image = cv2.resize(image, (new_w, new_h))
        scales = np.array([new_w / w, new_h / h])
        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            RandomResizeScale(size=self.size, ratio=(3. / 4, 5. / 2)),
            RandomCropFlip(),
            RandomResizedCrop(),
            RotatePadding(up=45, colors=True),
            # RandomResizePadding(size=self.size, random_scale=self.input_scale),
            ResizeLimitSquare(size=self.size),
            # RandomBrightness(),
            # RandomContrast(),
            RandomMirror(),
            Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            # Resize(size),
            ResizeSquare(size=self.size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransformNresize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)
