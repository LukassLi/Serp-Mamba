import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from augmentations.ctaugment import OPS
import augmentations
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image


# all sample
class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            ops_weak=None,
            ops_strong=None,
    ):
        self._base_dir = base_dir
        print(self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_image/")

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val_image/")

        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/test_image/")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image = np.array(Image.open(self._base_dir + "/training_image/{}".format(case), "r").convert('L'))
            # label = np.array(Image.open(self._base_dir + "/training_label/{}".format(case).replace('jpg', 'tif').replace('JPG', 'tif'), "r"))
            label = np.array(Image.open(
                self._base_dir + "/training_label/{}".format(case).replace('Img', 'Label').replace('tif', 'png'), "r"))
            if len(label.shape) == 3:
                label = label.mean(axis=-1)
            label[label > 0] = 1


        elif self.split == "val":
            image = np.array(Image.open(self._base_dir + "/val_image/{}".format(case), "r").convert('L'))
            label = np.array(Image.open(
                self._base_dir + "/val_label/{}".format(case).replace('Img', 'Label').replace('tif', 'png'), "r"))

            if len(label.shape) == 3:
                label = label.mean(axis=-1)
            label[label > 0] = 1


        elif self.split == "test":
            image = np.array(Image.open(self._base_dir + "/test_image/{}".format(case), "r").convert('L'))
            label = np.array(Image.open(
                self._base_dir + "/test_label/{}".format(case).replace('Img', 'Label').replace('tif', 'png'), "r"))

            if len(label.shape) == 3:
                label = label.mean(axis=-1)
            label[label > 0] = 1

        if self.split == "train":
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
            sample["name"] = case
        else:
            sample = {"image": image, "label": label.astype(np.int16)}
            sample["name"] = case

        sample["idx"] = idx
        return sample



def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        if len(label.shape) == 2:
            label = np.rot90(label, k)
            label = np.flip(label, axis=axis).copy()
            return image, label
        elif len(label.shape) == 3:
            new_label = np.zeros_like(label)
            for i in range(new_label.shape[0]):
                new_label[i, ...] = np.rot90(label[i, ...], k)
                new_label[i, ...] = np.flip(label[i, ...], axis=axis).copy()
            return image, new_label
        else:
            Exception("Error")
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
    elif len(label.shape) == 3:
        new_label = np.zeros_like(label)
        for i in range(label.shape[0]):
            new_label[i, ...] = ndimage.rotate(
                label[i, ...], angle, order=0, reshape=False)
        return image, new_label
    else:
        Exception("Error")


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # label = label.mean(axis=-1)  # 将颜色通道转换为灰度
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        # # more aug
        # if random.random() > 0.5:
        #     image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        # print(image.shape)
        x, y = image.shape

        # print(label.shape)
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # print(image.shape)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {"image": image, "label": label}
        return sample


class RandomGenerator2(object):
    def __init__(self, output_size):
        self.output_size = [256, 256]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape

        # image = zoom(
        #     image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # label = zoom(
        #     label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # print("image.shape=",image.shape)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class RandomGenerator_Multi_Rater(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        x, y = image.shape

        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(label.shape) == 2:
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(label.shape) == 3:
            label = zoom(
                label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample



def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
