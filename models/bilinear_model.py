import torch.nn as nn
import torch
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from torchvision.models import resnet34 as resnet
from .DieT import deit_small_patch16_224 as deit
from .cait_fuse import Cait_fuse, Cait_fusev2
import math
from .models_to_finetune import myresnetv2_task2

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class TransFuse_S(nn.Module):
    def __init__(self, num_classes=320, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_S, self).__init__()

        #self.resnet = resnet()
        self.resnet = myresnetv2(freeze=True)


        #self.transformer = deit(pretrained=pretrained, freeze = True)
        self.transformer = Cait_fuse(freeze=True)

        self.final_x = nn.Sequential(
            Conv(256, 512 , 3,  bn=True, relu=True),
            Conv(512, 1024, 3, stride=2, bn=True, relu=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.up_c = BiFusion_block(ch_1=256, ch_2=192, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)


        self.drop = nn.Dropout2d(drop_rate)
        self.head = nn.Linear(in_features=1024, out_features=num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((14,14))
        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, cls_token = self.transformer(imgs)  # N X num_patches X embed_dim
        x_b = torch.transpose(x_b, 1, 2) # N X embed_dim X num_patches
        x_b = x_b.view(x_b.shape[0], -1, int(math.sqrt(x_b.shape[2])), int(math.sqrt(x_b.shape[2])))  # change done here N X 384 X 14 X 14
        x_b = self.drop(x_b)   # N X embed_dim X sqrt(num_patches) X sqrt(num_patches)
        x_b = self.avgpool(x_b)

        x_u = self.resnet(imgs)
        x_u = self.avgpool(x_u)
        # joint path
        x_c = self.up_c(x_u, x_b)   # N X 256 X 14 X 14

        x = self.final_x(x_c)
        x = x.reshape((x.shape[0], -1))
        return self.head(x)


    def init_weights(self):
        self.final_x.apply(init_weights)

        self.up_c.apply(init_weights)

#############################################
class TransFuse_Sv2(nn.Module):
    def __init__(self, num_classes=320, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_Sv2, self).__init__()

        #self.resnet = resnet()
        self.transformer = Cait_fusev2(freeze=True)

        self.resnet = myresnetv2_v2(freeze=True)


        self.final_x = nn.Sequential(
            Conv(256, 512 , 3,  bn=True, relu=True),
            Conv(512, 1024, 3, stride=2, bn=True, relu=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.up_c = BiFusion_block(ch_1=256, ch_2=192, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)


        self.drop = nn.Dropout2d(drop_rate)
        self.head = nn.Linear(in_features=1024, out_features=num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((14,14))
        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b, cls_token = self.transformer(imgs)  # N X num_patches X embed_dim
        x_b = torch.transpose(x_b, 1, 2) # N X embed_dim X num_patches
        x_b = x_b.view(x_b.shape[0], -1, int(math.sqrt(x_b.shape[2])), int(math.sqrt(x_b.shape[2])))  # change done here N X 384 X 14 X 14
        x_b = self.drop(x_b)   # N X embed_dim X sqrt(num_patches) X sqrt(num_patches)
        x_b = self.avgpool(x_b)

        x_u = self.resnet(imgs)
        x_u = self.avgpool(x_u)
        # joint path
        x_c = self.up_c(x_u, x_b)   # N X 256 X 14 X 14

        x = self.final_x(x_c)
        x = x.reshape((x.shape[0], -1))
        return self.head(x)


    def init_weights(self):
        self.final_x.apply(init_weights)

        self.up_c.apply(init_weights)




#############################################

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class MyResnetv2_fusion(nn.Module):
    """
    Using resnetv2 from timm library, Added another fc layer to project features to 512-dim before
    passing to the classification head.
    """

    def __init__(self, freeze_layers=True):
        super(MyResnetv2_fusion, self).__init__()
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model.head = nn.Identity()
        if freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False
        self.model = model
        self.conv = nn.Sequential(
             nn.Conv2d(1024, 256, 1),
             nn.BatchNorm2d(256),
             nn.ReLU()
        )

    def forward(self, x):
        x= self.model.stem(x)
        for idx, stage in enumerate(self.model.stages):

            x = stage(x)
            if idx == 2:
                break
        return  self.conv(x)

def myresnetv2(freeze =True):
    model = MyResnetv2_fusion(freeze_layers= freeze)
    return model



#######################################################################################
class MyResnetv2_fusion_finetune_pretraining(nn.Module):
    """
    Using resnetv2 from timm library, Added another fc layer to project features to 512-dim before
    passing to the classification head.
    """

    def __init__(self, freeze_layers=False):
        super(MyResnetv2_fusion_finetune_pretraining, self).__init__()
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model.head = nn.Identity()
        if freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False

        self.model = model

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Flatten(),
                                   nn.Linear(in_features=2048, out_features=320),
                                   )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1024, 256, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


def myresnetv2_finetune_pretraining(freeze =False):
    model = MyResnetv2_fusion_finetune_pretraining(freeze_layers= freeze)
    return model



#########################################################################################


##############################################################

class MyResnetv2_fusionv2(nn.Module):
    """
    Using resnetv2 from timm library, Added another fc layer to project features to 512-dim before
    passing to the classification head.
    """

    def __init__(self, freeze_layers=True):
        super(MyResnetv2_fusionv2, self).__init__()
        model = myresnetv2_finetune_pretraining()
        path = "/home/hashmat.malik/Fall 2021/CV703 Lab/Week5/datasets/modelresnetv2-384_model_training_for_fusion_model_best.pth.tar"
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        model.head = nn.Identity()
        if freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False
        self.model = model
        self.conv = nn.Sequential(
             nn.Conv2d(1024, 256, 1),
             nn.BatchNorm2d(256),
             nn.ReLU()
        )

    def forward(self, x):
        x= self.model.model.stem(x)
        for idx, stage in enumerate(self.model.model.stages):

            x = stage(x)
            if idx == 2:
                break
        return  self.conv(x)

def myresnetv2_v2(freeze =True):
    model = MyResnetv2_fusionv2(freeze_layers= freeze)
    return model


############################################################
if __name__ =="__main__":
    img = torch.randn(1, 3, 384, 384)
    model = TransFuse_Sv2()
    out = model(img)
    print(out.shape)