# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from .helpers import get_chip_from_img


class PluginDataset(Dataset):
    """Dataset to load animal data for inference.
    Used in plugin.
    """

    def __init__(
        self,
        chips,
        image_paths,
        names,
        bboxes,
        viewpoints,
        thetas,
        transform,
        fliplr=False,
        fliplr_view=None,
        crop_bbox=True,
        use_chips=True
    ):
        self.chips = chips
        self.image_paths = image_paths
        self.bboxes = bboxes
        self.names = names
        self.transform = transform
        self.viewpoints = viewpoints
        self.crop_bbox = crop_bbox
        self.use_chips = use_chips

        thetas = [t if t is not None else 0 for t in thetas]
        self.thetas = thetas

        if fliplr:
            assert isinstance(fliplr_view, list) and all(
                isinstance(item, str) for item in fliplr_view
            )

        self.fliplr = fliplr
        self.fliplr_view = fliplr_view

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if len(image.shape) == 2: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        if self.use_chips:
            image = self.chips[idx]
        else:
            image_path = self.image_paths[idx]
            bbox = self.bboxes[idx]
            theta = self.thetas[idx]

            image = self.load_image(image_path)
            if image is None:
                raise ValueError('Fail to read {}'.format(self.image_paths[id]))

            # # Crop bounding box area
            # if  self.crop_bbox:
            #     x1, y1, w, h = bbox
            #     image = image[y1 : y1 + h, x1 : x1 + w]
            #     if min(image.shape) < 1:
            #         # Use original image
            #         image = self.load_image(image_path)
            #         self.bboxes[idx] = [0, 0, image.shape[1], image.shape[0]]
            
            # Crop a rotated bbox
            if self.crop_bbox:
                image = get_chip_from_img(image, bbox, theta)

        if self.fliplr:
            if self.viewpoints[idx] in self.fliplr_view:
                image = np.fliplr(image)

        if self.transform is not None:
            augmented = self.transform(image=image.copy())
            image = augmented['image']
            # image = self.transform(image.copy())
            
        return image, self.names[idx], self.image_paths[idx], torch.Tensor(self.bboxes[idx]), self.thetas[idx]