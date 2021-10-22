import timm
import torch.nn as nn
from datasets import dog_dataset, cub_dataset, food_dataset, cub_and_dogs
from models.models_to_finetune import deit_small_patch16_224, myresnet, myresnetv2_for_c_loss
import PIL
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
import config
import sys
import math
from run_center_loss import train_model_with_closs
from loss import CenterLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
batch_size = 56
transform = [config.data_transform, config.data_transform1, config.data_transform2, config.data_transform3, config.data_transform5]


# Load Model
count = 0

for tf in transform:
# for lr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 0.9]:
    # Load Data
    for lr in [0.01, 0.001, 0.005, 0.1]:
        train_loader, val_loader, test_loader = cub_and_dogs(bs=batch_size, data_transform=tf)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = myresnetv2_for_c_loss(num_classes=320)
        #model = deit_small_patch16_224(pretrained=True, use_top_n_heads=12, use_patch_outputs=False)
        #model = timm.create_model('resnet34', pretrained=True)
        #model = myresnet(num_classes=200)




        #model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=200)  # dogs dataset has 120 classes
        #model.head.apply(model._init_weights)
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = True

        my_list = ['head.1.weight', 'head.1.bias','head.3.weight', 'head.3.bias']
        params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))

        crit_entr = torch.nn.CrossEntropyLoss()
        crit_closs = CenterLoss(num_classes=320, feat_dim=512)
        # path = "/home/u20020019/Fall 2021/CV703 Lab/Week5/datasets/model_best_resnet_v2_cubs_dogs_0.pth.tar"
        # checkpoint = torch.load(path)
        # model.load_state_dict(checkpoint['state_dict'])
        optimizer = optim.Adam([{'params':  [i[1]for i in params], 'lr': 0.0001, 'betas': (0.5, 0.999)},
                        {'params':  [i[1]for i in base_params], 'lr': 0.00001, 'betas': (0.5, 0.999)},
                        {'params': crit_closs.parameters(), 'lr': lr, 'betas': (0.5, 0.999)}
                                ])

        scheduler = ReduceLROnPlateau(optimizer, 'max')
        train_model_with_closs(60, train_loader, val_loader, test_loader, optimizer,scheduler, crit_entr, crit_closs, model, f'resnet_v2_closs_new_lr_{lr}', is_train=True)
        count += 1
    break
# 0.5 loss goes to 158 for 1 epoch
# 0.2 central loss goes to 43 for 1 epoch and 17 for 2 epoch
# 0.01 1e-3
        # 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