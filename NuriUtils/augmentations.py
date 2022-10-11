import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from PIL import ImageFilter
import random


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

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32)


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, images):
        images_ = []
        for image in images:
            if random.randint(2):
                images_.append(image)
            else:
                height, width, depth = image.shape
                ratio = random.uniform(1, 4)
                left = random.uniform(0, width*ratio - width)
                top = random.uniform(0, height*ratio - height)
                expand_image = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image.dtype)
                expand_image[:, :, :] = self.mean
                expand_image[int(top):int(top + height), int(left):int(left + width)] = image
                images_.append(expand_image)
        images_ = np.stack(images_, 0)
        return images_


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, images):
        images_ = []
        for image in images:
            im = image.copy()
            im = self.rand_brightness(im)
            if random.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            im = distort(im)
            im = self.rand_light_noise(im)
            images_.append(im)
        try:
            images_ = np.stack(images_, 0)
        except:
            print("aa")
        return images_


class Resize(object):
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, images):
        images_ = []
        for image in images:
            images_.append(cv2.resize(image, (self.size[1], self.size[0])))
        images_ = np.stack(images_, 0)
        return images_


class AddBboxPatch(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, depth, bboxes):
        if random.random() < self.p:
            H, W, C = image.shape
            bboxes_ = bboxes.clone()
            bbox_size = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            size_mask = (bbox_size < 0.7) & (bbox_size > 0.1)
            bboxes_ = bboxes_[size_mask].reshape(-1, 4)
            if len(bboxes_) > 0:
                bboxes_[:, 0] = (bboxes_[:, 0] * W)
                bboxes_[:, 1] = (bboxes_[:, 1] * H)
                bboxes_[:, 2] = (bboxes_[:, 2] * W)
                bboxes_[:, 3] = (bboxes_[:, 3] * H)
                bboxes_ = bboxes_.to(torch.int32)
                i = np.random.choice(len(bboxes_))
                image[bboxes_[i, 1]:bboxes_[i, 3], bboxes_[i, 0]:bboxes_[i, 2]] = 0
                depth[bboxes_[i, 1]:bboxes_[i, 3], bboxes_[i, 0]:bboxes_[i, 2]] = 0
        return image, depth


class ReplaceBboxPatch(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, bboxes, depth):
        if random.random() < self.p:
            H, W, C = image.shape
            bboxes_ = bboxes.copy()
            bbox_size = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            size_mask = (bbox_size < 0.7 * H * W)
            bboxes_ = bboxes_[size_mask].reshape(-1, 4)
            if random.random() < 0.5:
                if len(bboxes_) >= 2:
                    bboxes_ = bboxes_.astype(np.int32)
                    i, j = np.random.choice(len(bboxes_), 2, replace=False)
                    bbox1_rgb = image[bboxes_[i, 1]:bboxes_[i, 3], bboxes_[i, 0]:bboxes_[i, 2]].copy()
                    bbox2_rgb = image[bboxes_[j, 1]:bboxes_[j, 3], bboxes_[j, 0]:bboxes_[j, 2]].copy()
                    image[bboxes_[i, 1]:bboxes_[i, 3], bboxes_[i, 0]:bboxes_[i, 2]] = cv2.resize(bbox2_rgb, (bbox1_rgb.shape[1], bbox1_rgb.shape[0]))
                    image[bboxes_[j, 1]:bboxes_[j, 3], bboxes_[j, 0]:bboxes_[j, 2]] = cv2.resize(bbox1_rgb, (bbox2_rgb.shape[1], bbox2_rgb.shape[0]))
                    return image, i, j, True, depth
            else:
                if len(bboxes_) > 0:
                    bboxes_ = bboxes_.astype(np.int32)
                    i = np.random.choice(len(bboxes_))
                    image[bboxes_[i, 1]:bboxes_[i, 3], bboxes_[i, 0]:bboxes_[i, 2]] = 0
                    depth[bboxes_[i, 1]:bboxes_[i, 3], bboxes_[i, 0]:bboxes_[i, 2]] = 0
        return image, None, None, False, depth


class TrainAugmentation(object):
    def __init__(self, size=(224, 224), mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # Resize(size),
            ConvertFromInts(),
            PhotometricDistort(),
            SubtractMeans(self.mean),
            Resize(size)
        ])

    def __call__(self, img):
        return self.augment(img)


class OnlyResize(object):
    def __init__(self, size=(224, 224)):
        self.size = size
        self.augment = Compose([
            Resize(size)
        ])

    def __call__(self, img):
        return self.augment(img)


class BasicAugmentation(object):
    def __init__(self, size=(224,224), mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            SubtractMeans(self.mean),
            Resize(size)
        ])

    def __call__(self, img):
        return self.augment(img)


class ResizeAugmentation(object):
    def __init__(self, size=(224, 224)):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            Resize(size)
        ])

    def __call__(self, img):
        return self.augment(img)


class ShowAugmentation(object):
    def __init__(self, size=(224, 224), mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # ConvertFromInts(),
            # SubtractMeans(self.mean),
            Resize(size)
        ])

    def __call__(self, img):
        return self.augment(img)


class DetectTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.augment = Compose([
            Resize(size)
        ])

    # assume input is cv2 img for now
    def __call__(self, img):
        x = self.augment(img)
        x = x.astype(np.float32)
        x -= self.mean
        return x

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
