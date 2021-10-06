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
import timm
epochs = 10
batch_size = 256
transform = config.data_transform
# Load Data
train_loader, val_loader, test_loader = cub_dataset(bs=batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Model
model = timm.create_model("convit_tiny", pretrained=True)


            
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
# model = deit_small_patch16_224(pretrained=True, use_top_n_heads=12, use_patch_outputs=False)
model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=200)  # dogs dataset has 120 classes
model.head.apply(model._init_weights)
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
    
for name,param in model.named_parameters():
        if name == "head.weight" or "head.bias":
            param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()

train_model(25, train_loader, val_loader, optimizer, criterion, model, 'convit')



# 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