import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def train_model_with_closs(epochs, train_loader, val_loader, test_loader, optimizer, scheduler, crit_ent, crit_closs,  model, arch, is_train=True):
    best_acc1 = 0

    if not is_train:
        #a1 = validate(val_loader, model, crit_ent, crit_closs)
        a2 = validate(test_loader, model, crit_ent, crit_closs)
    else:
        for epoch in range(epochs):

            #adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, crit_ent, crit_closs, optimizer, epoch)

            # evaluate on validation set
            acc1 = validate(val_loader, model, crit_ent, crit_closs)
            scheduler.step(acc1)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)


            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, crit_ent, crit_closs, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    entropy_losses = AverageMeter('ent_Loss', ':.4e')
    c_losses = AverageMeter('center_loss', ':.4e')
    losses = AverageMeter('loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, entropy_losses, c_losses, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        features, output = model(images)

        loss_ent = crit_ent(output, target)
        loss_center = crit_closs(features, target)
        # total_loss = loss_ent + 0.003 * loss_center
        total_loss = loss_ent + 0.003 * loss_center
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        entropy_losses.update(loss_ent.item(), images.size(0))
        losses.update(total_loss.item(), images.size(0))
        c_losses.update(loss_center.item(), images.size(0))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            progress.display(i)


def validate(val_loader, model, crit_ent, crit_closs):

    batch_time = AverageMeter('Time', ':6.3f')
    entropy_losses = AverageMeter('ent_Loss', ':.4e')
    c_losses = AverageMeter('center_loss', ':.4e')
    losses = AverageMeter('loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, entropy_losses, c_losses, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        end = time.time()
        acc = 0
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            features, output = model(images)

            loss_ent = crit_ent(output, target)
            loss_center = crit_closs(features, target)
            # total_loss = loss_ent + 0.003 * loss_center
            total_loss = loss_ent + 0.003 * loss_center
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            entropy_losses.update(loss_ent.item(), images.size(0))
            losses.update(total_loss.item(), images.size(0))
            c_losses.update(loss_center.item(), images.size(0))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5== 0:
                progress.display(i)
        #print('Accuracy of val set:{0:.3%}'.format(acc / (len(val_loader)*target.shape[0])))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='./datasets/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"./datasets/model_best_{state['arch']}.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        #pred = pred.t()
        #correct = pred.eq(target.view(1, -1).expand_as(pred))
        target = target.reshape((target.shape[0], 1))
        target = target.repeat(1, maxk)
        correct = pred.eq(target)

        res = []
        for k in topk:
            correct_k = correct[:, 0:k].reshape((-1)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

