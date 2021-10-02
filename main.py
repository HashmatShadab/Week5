from datasets import dog_dataset, cub_dataset, food_dataset
from models.models_to_finetune import deit_small_patch16_224
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
from vit.vit_pytorch.cct import CCT
epochs = 10
batch_size = 192
transform = config.data_transform
# Load Data
train_loader, val_loader = dog_dataset(bs=batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Model
model = CCT(
        img_size=224,
        embedding_dim=384,
        n_conv_layers=2,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=14,
        num_heads=6,
        mlp_radio=3.,
        num_classes=120,
        positional_embedding='learnable', # ['sine', 'learnable', 'none']
        )

model = model.to(device)
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
# model = deit_small_patch16_224(pretrained=True, use_top_n_heads=12, use_patch_outputs=False)
# model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=120)  # dogs dataset has 120 classes
# model.head.apply(model._init_weights)
# model = model.to(device)
# for param in model.parameters():
#     param.requires_grad = True

# my_list = ['head.weight', 'head.bias']
# params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
# base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))

# param_list = []
# for name in base_params:
#     param_list.append((name[0], name[1]))

# block11 = [name[1]   for name in param_list if 'blocks.11' in name[0]]
# block11.extend([param_list[-2][1],param_list[-1][1]])
# block10 = [name[1]   for name in param_list if 'blocks.10' in name[0]]
# block9 = [name[1]   for name in param_list if 'blocks.9' in name[0]]
# block8 = [name[1]   for name in param_list if 'blocks.8' in name[0]]

# optimizer = optim.Adam([{'params': block11, 'lr':0.001, 'betas': (0.5, 0.999)},
#                 {'params': block10, 'lr':0.0001, 'betas': (0.5, 0.999)},
#                 {'params': block9, 'lr': 0.0001, 'betas': (0.5, 0.999)},
#                 {'params': block8, 'lr': 0.00001, 'betas': (0.5, 0.999)},
#                 {'params':  [i[1]for i in params], 'lr': 0.001, 'betas': (0.5, 0.999)}])

criterion = torch.nn.CrossEntropyLoss()

train_model(10, train_loader, val_loader, optimizer, criterion, model, 'CCT')



# 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