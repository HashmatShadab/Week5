import torch
import torch.nn as nn




class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


"""
Credits: Dr. Salman @ salman.khan@mbzuai.ac.ae
Implemented by : Mohammad Maaz :  https://github.com/mmaaz60/ssl_for_fgvc/blob/main/loss/gb_loss.py
"""

import torch
import torch.nn as nn


class GBLoss(torch.nn.Module):
    """
    The class implements the gradient-boosting (GB) loss introduced in
    "Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes".
    (http://arxiv.org/abs/1912.06842).
    """
    def __init__(self):
        """
        Constructor, initialize the base cross-entropy loss
        """
        # Call the parent constructor
        super(GBLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()  # Cross entropy loss

    def forward(self, x, y):
        """
        The function implements the forward pass for the GB loss.
        :param x: Predictions
        :param y: Ground truth labels
        """
        x1 = x.clone()
        x1[range(x1.size(0)), y] = -float("Inf")
        x_gt = x[range(x.size(0)), y].unsqueeze(1)
        x_topk = torch.topk(x1, 15, dim=1)[0]  # 15 Negative classes to focus on, its a hyperparameter
        x_new = torch.cat([x_gt, x_topk], dim=1)

        return self.ce(x_new, torch.zeros(x_new.size(0)).cuda().long())