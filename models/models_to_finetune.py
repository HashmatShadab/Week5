from functools import partial
import timm
import torch
import torch.nn as nn
import torchvision
from einops import reduce
from timm.models.vision_transformer import VisionTransformer, _cfg


class InferenceVisionTransformer(VisionTransformer):
    def __init__(self, use_top_n_heads=1, use_patch_outputs=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches = use_patch_outputs
        self.top_n_heads = use_top_n_heads
        if self.patches:
            self.head = torch.nn.Linear(self.embed_dim * (self.top_n_heads + 1), self.num_classes)
        else:
            self.head = torch.nn.Linear(self.embed_dim * self.top_n_heads, self.num_classes)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        layer_wise_tokens = []
        for blk in self.blocks:
            x = blk(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]

        return [x[:, 0] for x in layer_wise_tokens], [x for x in layer_wise_tokens]

    def forward(self, x, patches=False, fine_tune=False, get_feat=False):
        if fine_tune:
            with torch.no_grad():
                list_out, patch_out = self.forward_features(x)
            if self.patches:
                patch_avg = reduce(patch_out[-1][:, 1:], "B P C -> B C", reduction="mean")
                joint_feature = torch.cat(list_out[-self.top_n_heads:] + [patch_avg, ], dim=-1)
            else:
                joint_feature = torch.cat(list_out[-self.top_n_heads:], dim=-1)
            if get_feat:
                return joint_feature
            return self.head(joint_feature)
        else:
            list_out, patch_out = self.forward_features(x)
            if self.patches:
                patch_avg = reduce(patch_out[-1][:, 1:], "B P C -> B C", reduction="mean")
                joint_feature = torch.cat(list_out[-self.top_n_heads:] + [patch_avg, ], dim=-1)
            else:
                joint_feature = torch.cat(list_out[-self.top_n_heads:], dim=-1)
            if get_feat:
                return joint_feature
            return self.head(joint_feature)


class InferenceResnet(torchvision.models.ResNet):
    def _forward_impl(self, x, get_feat=False, fine_tune=False):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if get_feat:
            return x, self.fc(x)
        x = self.fc(x)

        return x

    def forward(self, x, get_feat=False, fine_tune=False):
        return self._forward_impl(x, get_feat=get_feat, fine_tune=fine_tune)


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = InferenceVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        msg = model.load_state_dict({x: y for x, y in checkpoint["model"].items() if x not in ["head.weight",
                                                                                               "head.bias"]},
                                    strict=False)
        print(msg)
    return model


def deit_small_patch16_224(pretrained=False, **kwargs):
    model = InferenceVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        msg = model.load_state_dict({x: y for x, y in checkpoint["model"].items() if x not in ["head.weight",
                                                                                               "head.bias"]},
                                    strict=False)
        print(msg)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    arch = 'resnet18'
    block = torchvision.models.resnet.BasicBlock
    layers = [2, 2, 2, 2]
    model = InferenceResnet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    arch = 'resnet50'
    block = torchvision.models.resnet.Bottleneck
    layers = [3, 4, 6, 3]
    model = InferenceResnet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

class MyResnetv2_c_loss(nn.Module):
    """
    Using resnetv2 from timm library, Added another fc layer to project features to 512-dim before
    passing to the classification head.
    """

    def __init__(self, num_classes):
        super(MyResnetv2_c_loss, self).__init__()
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model.head = nn.Identity()
        self.feature = nn.Sequential(model,
                                     nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Flatten(),
                                     nn.Linear(in_features=2048, out_features=512),
                                     nn.ReLU(),
                                     )
        #self.feature = model
        self.head = nn.Sequential(

                                     nn.Linear(in_features=512, out_features=num_classes)
                                     )

    def forward(self, x):
        features = self.feature(x)
        x = self.head(features)
        return features, x

def myresnetv2_for_c_loss(num_classes=200):
    model = MyResnetv2_c_loss(num_classes=num_classes)
    return model


class MyResnetv2_task2(nn.Module):

    def __init__(self, num_classes):
        super(MyResnetv2_task2, self).__init__()
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model.head = nn.Identity()
        self.feature = nn.Sequential(model,
                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten(),
                                     )
        # self.feature = model
        self.head = nn.Sequential(

            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        features = self.feature(x)
        x = self.head(features)
        return x


def myresnetv2_task2(num_classes=320):
    model = MyResnetv2_task2(num_classes=num_classes)
    return model


class MyResnetv2_task1(nn.Module):

    def __init__(self, num_classes):
        super(MyResnetv2_task1, self).__init__()
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model.head = nn.Identity()
        self.backbone = nn.Sequential(model,
                                      nn.AdaptiveAvgPool2d((1, 1))
                                      )

        # self.feature = model
        self.head = nn.Sequential(nn.Flatten(),

                                  nn.Linear(in_features=2048, out_features=num_classes)
                                  )

    def forward(self, x):
        features = self.backbone(x)
        x = x.reshape((x.shape[0], -1))
        x = self.head(features)
        return x


def myresnetv2_task1(num_classes=200):
    model = MyResnetv2_task1(num_classes=num_classes)
    return model


if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224)
    model = myresnetv2()
    output = model(img)