import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask[None, :]  # adding channel dimention


def dice_loss(true: np.ndarray, predicted: np.ndarray, smooth=1e-6):
    true = true.flatten()
    predicted = predicted.flatten()
    intersection = np.sum(true * predicted)
    dice_coefficient = (2 * intersection + smooth) / \
        (true.sum() + predicted.sum() + smooth)
    return 1 - dice_coefficient
