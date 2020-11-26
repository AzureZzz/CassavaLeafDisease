from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, path, transforms=None, mode='train'):
        self.transforms = transforms
        self.img_path = os.path.join(path, 'train_images')
        self.data_list = self._get_data_list(path, mode)

    def _get_data_list(self, path, mode):
        df = pd.read_csv(os.path.join(path, 'train.csv'))
        data_list = [(x, y) for (x, y) in zip(df[df.keys()[0]], df[df.keys()[1]])]
        data_list = shuffle(data_list)
        train_list, val_list = train_test_split(data_list, test_size=0.3, random_state=41)
        if mode == 'train':
            return train_list
        else:
            return val_list

    def __getitem__(self, idx):
        img_name = self.data_list[idx][0]
        label = self.data_list[idx][1]
        image = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        image = np.array(image)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        return image, label

    def __len__(self):
        return len(self.data_list)


def get_dataset(path, batch_size, batch_size_val, train_transforms, val_transforms):
    train_dataset = MyDataset(path, train_transforms, 'train')
    val_dataset = MyDataset(path, val_transforms, 'val')
    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size_val)
    return train_loader, val_loader