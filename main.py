from datasets import dog_dataset, cub_dataset, food_dataset,cub_and_dogs
from models.models_to_finetune import deit_small_patch16_224, myresnetv2_task1, myresnetv2_task2, myresnetv2_for_c_loss
from models import bilinear_model
import PIL
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
import config
import sys
import math
from run import train_model
from vit.vit_pytorch.nest import NesT
import timm
from loss import GBLoss

epochs = 30
batch_size = 96
torch.manual_seed(42)
test_transform=transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
transform = transforms.Compose([  # Accuracy:87.622%

        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# Load Data
train_loader, val_loader,test_loader = cub_and_dogs(bs=batch_size, data_transform=transform, test_transform=test_transform)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = bilinear_model.TransFuse_Sv2()

path = "/home/hashmat.malik/Fall 2021/CV703 Lab/Week5/datasets/modelfusion_model_with_pretraining_from_dataset_with_gbloss_best.pth.tar"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
#
# my_list = ['head.weight', 'head.bias']
# params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
# base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))


#
# optimizer = optim.Adam([
#                 {'params':  [i[1]for i in params], 'lr': 0.0001, 'betas': (0.5, 0.999)},
#                 {'params':  [i[1]for i in base_params], 'lr': 0.00001, 'betas': (0.5, 0.999)}])

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

#criterion = torch.nn.CrossEntropyLoss()
criterion = GBLoss()
train_model(epochs, train_loader, val_loader, test_loader, optimizer, criterion, model, 'fusion_model_with_pretraining_from_dataset_with_gbloss', is_train=False)



# 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