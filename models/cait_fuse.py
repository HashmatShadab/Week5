import torch
import torch.nn as nn
import timm


class Cait_fuse(nn.Module):
    def __init__(self, freeze=True):
        super(Cait_fuse, self).__init__()
        model = timm.create_model("cait_xxs24_384", pretrained=True)
        # model.head = torch.nn.Linear(in_features=model.head.in_features,
        #                              out_features=200)  # dogs dataset has 120 classes
        # model.head.apply(model._init_weights)
        model.head = nn.Identity()
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        self.model = model

        #path = "/home/hashmat.malik/Fall 2021/CV703 Lab/Week5/datasets/Task1:cub_dataset_weights/Exp3/modelcait_xxs24_3845_best.pth.tar"
        # checkpoint = torch.load(path)
        # model.load_state_dict(checkpoint['state_dict'])
        # self.patch_embed = model.patch_embed
        # self.pos_drop = model.pos_drop
        # self.sa = nn.Sequential(model.blocks)
    def forward_features(self, x):
        B = x.shape[0]
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for i, blk in enumerate(self.model.blocks):
            x = blk(x)

        for i, blk in enumerate(self.model.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.model.norm(x)
        return x



    def forward(self, x):

        x = self.forward_features(x)
        cls_token, feat = x[:, 0], x[:, 1:]
        return feat,cls_token

##################################
class Cait_fusev2(nn.Module):
    def __init__(self, freeze=True):
        super(Cait_fusev2, self).__init__()
        model = timm.create_model("cait_xxs24_384", pretrained=True)
        model.head = torch.nn.Linear(in_features=model.head.in_features,
                                      out_features=320)
        model.head.apply(model._init_weights)
        path = "/home/hashmat.malik/Fall 2021/CV703 Lab/Week5/datasets/modelcait_xxs24_384_task2exp7_for_nb_best.pth.tar"
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        self.model = model

        #path = "/home/hashmat.malik/Fall 2021/CV703 Lab/Week5/datasets/Task1:cub_dataset_weights/Exp3/modelcait_xxs24_3845_best.pth.tar"
        # checkpoint = torch.load(path)
        # model.load_state_dict(checkpoint['state_dict'])
        # self.patch_embed = model.patch_embed
        # self.pos_drop = model.pos_drop
        # self.sa = nn.Sequential(model.blocks)
    def forward_features(self, x):
        B = x.shape[0]
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for i, blk in enumerate(self.model.blocks):
            x = blk(x)

        for i, blk in enumerate(self.model.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.model.norm(x)
        return x



    def forward(self, x):

        x = self.forward_features(x)
        cls_token, feat = x[:, 0], x[:, 1:]
        return feat,cls_token


#################################
if __name__ == "__main__":
    img = torch.randn(1, 3, 384, 384)
    model = Cait_fusev2()
    out = model(img)
    a = 2