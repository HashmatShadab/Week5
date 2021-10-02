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


class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
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
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

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
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            torchvision.transforms.functional.to_tensor(Image.open(row["path"])), row['label']
        )




def dog_dataset(data_root = "/home/u20020019/Fall 2021/CV703 Lab/Week5/dog/",
            data_transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]), bs=32):



    train_dataset = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    test_dataset = DOGDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")

    # load in into the torch dataloader to get variable batch size, shuffle
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, drop_last=False, shuffle=True)

    return train_loader, test_loader


def cub_dataset(data_root="/home/u20020019/TransFG/CUB_200_2011",
                data_transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]), bs=32):
    train_dataset = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    test_dataset = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, drop_last=False, shuffle=True)

    return train_loader, test_loader



def food_dataset(data_dir = "/home/u20020019/Fall 2021/CV703 Lab/Week5/food_dataset", bs=32
                 ):


    split = 'train'
    train_df = pd.read_csv(f'{data_dir}/{split}_labels.csv', names=['image_name', 'label'])
    train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

    split = 'val'
    val_df = pd.read_csv(f'{data_dir}/{split}_labels.csv', names=['image_name', 'label'])
    val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))
    train_dataset = FOODDataset(train_df)
    val_dataset = FOODDataset(val_df)

    # load in into the torch dataloader to get variable batch size, shuffle
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, drop_last=False, shuffle=True)

    return train_loader, val_loader


def cub_and_dogs(cub_root = "/home/u20020019/TransFG/CUB_200_2011",
                 dog_root = "/home/u20020019/Fall 2021/CV703 Lab/Week5/dog/",
                data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                    ])
                 ):



    train_dataset_cub = CUBDataset(image_root_path=f"{cub_root}", transform=data_transform, split="train")
    test_dataset_cub = CUBDataset(image_root_path=f"{cub_root}", transform=data_transform, split="test")


    train_dataset_dog = DOGDataset(image_root_path=f"{dog_root}", transform=data_transform, split="train")
    test_dataset_dog = DOGDataset(image_root_path=f"{dog_root}", transform=data_transform, split="test")


    train_dataloader = torch.utils.data.DataLoader(
                 torch.utils.data.ConcatDataset([train_dataset_cub, train_dataset_dog]),
                 batch_size=1, shuffle=True,
                 num_workers=1, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
                 torch.utils.data.ConcatDataset([test_dataset_cub, test_dataset_dog]),
                 batch_size=1, shuffle=True,
                 num_workers=1, pin_memory=True)

    return train_dataloader, test_dataloader
