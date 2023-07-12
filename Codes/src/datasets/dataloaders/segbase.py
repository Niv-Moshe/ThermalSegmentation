'''
Code Adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/segbase.py
'''
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, base_size=520, crop_size=480, logger=None):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.logger = logger
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.count = 0
        edge_radius = 7
        self.edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (edge_radius, edge_radius))

        if self.mode != 'testval' and self.mode != 'test':
            self.sizes = dict(zip(crop_size, base_size))

    def _val_sync_transform(self, img, mask, edge, crop_size=None):
        outsize = self.crop_size[0]
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        edge = edge.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask, edge = self._img_transform(img), self._mask_transform(mask), self._edge_transform(edge)
        return img, mask, edge

    def _sync_transform(self, img, mask, edge, crop_size=None):
        # np.array(img).shape = [H, W, channels=3]
        # img is PIL image, so it works like that with the transforms without reshaping (if tensor then needed).
        # random blur
        if random.random() < 0.5:
            gaussian_blur = transforms.GaussianBlur(kernel_size=9)
            img = gaussian_blur(img)
        # random zoom in
        if random.random() < 0.5:
            # scale is actually a random range to sample the crop before resizing
            crop_resize = transforms.RandomResizedCrop(size=list(np.array(img).shape)[:-1], scale=(0.6, 0.6))
            img = crop_resize(img)
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        # plt.imshow(img)
        # plt.savefig(f"/home/ilan/Desktop/Niv/ThermalSegmentation/try/img_{self.count}")
        self.count += 1
        if crop_size is None:
            crop_size = self.crop_size[0]
            base_size = self.base_size[0]
        else:
            base_size = self.sizes[crop_size]

        # random scale (short edge)
        short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        edge = edge.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask, edge = self._img_transform(img), self._mask_transform(mask), self._edge_transform(edge)
        return img, mask, edge

    def im2double(self, im_: np.ndarray) -> np.ndarray:
        info = np.iinfo(im_.dtype)  # Get the data type of the input image
        return im_.astype(np.float64) / info.max  # Divide all values by the largest possible value in the datatype

    def np2Tensor(self, tensor):
        if len(tensor.shape) == 3:
            np_transpose = np.ascontiguousarray(tensor.transpose((2, 0, 1)))
        else:
            np_transpose = np.ascontiguousarray(tensor[np.newaxis, :])
        tensor = torch.from_numpy(np_transpose).float()
        return tensor

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    def _edge_transform(self, edge):
        return np.array(edge).astype('int32')
    # [None, :, :]

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
