# from confounder_dataset import ConfounderDataset
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class CUBDataset(Dataset):
    def __init__(self, df, data_dir):

        self.metadata_df = df
        self.data_dir = data_dir

        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1

        # Map to groups
        self.n_groups = 4
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype('int')
        self._group_array = torch.LongTensor(self.group_array)
        self._y_array = torch.LongTensor(self.y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(dim=1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(dim=1).float()

        # reading image files
        self.filename_array = self.metadata_df['img_filename'].values
        self.eval_transform = get_transform_cub()

        
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        place = self.confounder_array[idx]
        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")

        img = self.eval_transform(img)

        return img, y, g, place

    def group_counts(self):
        return self._group_counts
    
    def class_counts(self):
        return self._y_counts
    

    def get_loader(self, batch_size, shuffle=False, pin_memory=False):
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        return loader

    # def get_splits(self, splits: list):
    #     subsets = {}
    #     for split in splits:
    #         mask = self.split_array == self.split_dict[split]
    #         indices = np.where(mask)[0]
    #         subsets[split] = Subset(self, indices)
    #     return subsets


def get_transform_cub():
    scale = 256.0 / 224.0
    target_resolution = (224, 224)  # for resnet

    # Resizes the image to a slightly larger square then crops the center.
    transform = transforms.Compose([
        transforms.Resize(
            (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform
