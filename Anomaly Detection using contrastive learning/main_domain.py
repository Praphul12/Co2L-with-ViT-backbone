from __future__ import print_function

import os
import copy
import sys
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, set_optimizer, save_model, load_model
from networks.mlp import SupConMLP
from losses_negative_only import SupConLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument('--mem_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--dataset', type=str, default='r-mnist')
    parser.add_argument('--temp', type=float, default=0.07)
    opt = parser.parse_args()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return opt

def set_model(opt):
    model = SupConMLP().to(opt.device)
    criterion = SupConLoss(temperature=opt.temp).to(opt.device)
    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    losses = AverageMeter()
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(opt.device), labels.to(opt.device)
        optimizer.zero_grad()
        features, _ = model(images, return_feat=True)
        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), labels.size(0))
        if (idx + 1) % 100 == 0:
            print(f'Train Epoch [{epoch}] Loss: {losses.avg:.4f}')
    return losses.avg

def main():
    opt = parse_option()
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    train_loader = ...  # Load dataset
    logger = SummaryWriter(log_dir='./logs')
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        logger.add_scalar('Loss/train', loss, epoch)
    save_model(model, optimizer, opt, opt.epochs, 'final_model_domain.pth')
if __name__ == '__main__':
    main()
