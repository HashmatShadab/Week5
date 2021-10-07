from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import scipy.io
from PIL import Image
import pandas as pd
from torch.utils.data.dataset import random_split
torch.manual_seed(42)

test_transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])


class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", concat=False, *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path
        self.concat = concat
        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split


    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        if self.concat:
            class_to_idx = {cls_name: i+120 for i, cls_name in enumerate(classes)}
        else:
            class_to_idx = {cls_name: i  for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content



class DOGDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for DOG Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0] for o in image_info]

        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0] for o in split_info]
        self.split_info = {}
        if split == 'train':
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split == 'test':
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"

        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)




    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        content = scipy.io.loadmat(file_path)
        return content['file_list']


class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img = Image.open(row["path"])
        label = torch.tensor(int(row["label"]))
        img = self.transform(img)


        return (img,label)

        # return (
        #     torchvision.transforms.functional.to_tensor(Image.open(row["path"])), row['label']
        # )




def dog_dataset(data_root = "./datasets/dog/",
            data_transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]), bs=32):



    train_dataset = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    test_dataset = DOGDataset(image_root_path=f"{data_root}", transform=test_transform, split="test")
    lengths = [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1)]
    torch.manual_seed(0)
    train_set, val_set = random_split(train_dataset, lengths)
    # load in into the torch dataloader to get variable batch size, shuffle
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, drop_last=False, shuffle=True)

    return train_loader, val_loader, test_loader


def cub_dataset(data_root="./datasets/CUB_200_2011",
                data_transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]), bs=32):

    train_dataset = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    lengths = [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1) + 1]
    torch.manual_seed(0)
    train_set, val_set = random_split(train_dataset, lengths)
    test_dataset = CUBDataset(image_root_path=f"{data_root}", transform=test_transform, split="test")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, drop_last=True, shuffle=True)
    val_loader =   torch.utils.data.DataLoader(val_set, batch_size=bs, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, drop_last=False, shuffle=True)

    return train_loader, val_loader, test_loader



def food_dataset(data_dir = "./datasets/food_dataset",
                 data_transform=transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                 ]),
                    bs=32
                 ):


    split = 'train'
    train_df = pd.read_csv(f'{data_dir}/{split}_labels.csv', names=['image_name', 'label'])
    train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

    split = 'val'
    val_df = pd.read_csv(f'{data_dir}/{split}_labels.csv', names=['image_name', 'label'])
    val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))
    train_dataset = FOODDataset(train_df, transform=data_transform)
    test_dataset = FOODDataset(val_df, transform=test_transform)

    lengths = [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1) + 1]
    torch.manual_seed(0)
    train_set, val_set = random_split(train_dataset, lengths)
    # load in into the torch dataloader to get variable batch size, shuffle
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, drop_last=False, shuffle=True)

    return train_loader, val_loader, test_loader





def cub_and_dogs(cub_root = "./datasets/CUB_200_2011",
                 dog_root = "./datasets/dog/",
                data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                    ]),
                 bs=32
                 ):






    train_dataset_dog = DOGDataset(image_root_path=f"{dog_root}", transform=data_transform, split="train")
    lengths = [int(len(train_dataset_dog) * 0.9), int(len(train_dataset_dog) * 0.1)]
    torch.manual_seed(0)
    train_set_dog, val_set_dog = random_split(train_dataset_dog, lengths)
    test_dataset_dog = DOGDataset(image_root_path=f"{dog_root}", transform=test_transform, split="test")

    train_dataset_cub = CUBDataset(image_root_path=f"{cub_root}", transform=data_transform, split="train", concat=True)
    lengths = [int(len(train_dataset_cub) * 0.9), int(len(train_dataset_cub) * 0.1) + 1]
    torch.manual_seed(0)
    train_set_cub, val_set_cub = random_split(train_dataset_cub, lengths)
    test_dataset_cub = CUBDataset(image_root_path=f"{cub_root}", transform=test_transform, split="test", concat=True)


    train_dataloader = torch.utils.data.DataLoader(
                 torch.utils.data.ConcatDataset([train_set_dog, train_set_cub]),
                 batch_size=bs, shuffle=True,
                 num_workers=1, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([val_set_dog, val_set_cub ]),
        batch_size=bs, shuffle=True,
        num_workers=1, pin_memory=True)


    test_dataloader = torch.utils.data.DataLoader(
                 torch.utils.data.ConcatDataset([test_dataset_dog, test_dataset_cub]),
                 batch_size=bs, shuffle=False,
                 num_workers=1, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cub_and_dogs()
    data, labels = next(iter(test_loader))
    a=2