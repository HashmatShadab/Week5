from datasets import dog_dataset, cub_dataset, food_dataset
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
epochs = 50
batch_size = 128
torch.manual_seed(42)
test_transform=transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
transform = config.data_transform2
# Load Data
train_loader, val_loader,test_loader = cub_dataset(bs=batch_size, data_transform=transform, test_transform=test_transform)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model =  NesT(
#     image_size = 224,
#     patch_size = 4,
#     dim = 96,
#     heads = 3,
#     num_hierarchies = 3,        # number of hierarchies
#     block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
#     num_classes = 10
# )
#model = myresnetv2_task2(num_classes=320)
model = bilinear_model.TransFuse_S()
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
    
for name,param in model.named_parameters():
        if name == "head.weight" or "head.bias":
            param.requires_grad = True


# Load Model
# model = deit_small_patch16_224(pretrained=True, use_top_n_heads=12, use_patch_outputs=False)
# model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=200)  # dogs dataset has 120 classes
# model.head.apply(model._init_weights)
# model = model.to(device)
# for param in model.parameters():
#     param.requires_grad = True

# my_list = ['head.weight', 'head.bias']
# params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
# base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))



# optimizer = optim.Adam([
#                 {'params':  [i[1]for i in params], 'lr': 0.001, 'betas': (0.5, 0.999)}])

optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.5, 0.999))

criterion = torch.nn.CrossEntropyLoss()

train_model(epochs, train_loader, val_loader, test_loader, optimizer, criterion, model, 'fusion_model_with_pretrained_weights_on_cub_and_dogs', is_train=False)



# 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