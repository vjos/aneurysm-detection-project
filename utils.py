from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from models.pointnet import feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

classes = ["vessel", "aneurysm"]

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_model(model, train, test, epochs=5, feat_trans=False):
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.to(dev)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}", end="\r")
        scheduler.step()
        for i, (pcld, label) in enumerate(train):
            pcld = pcld.transpose(2, 1)
            pcld, label = pcld.to(dev), label.to(dev)
            optimizer.zero_grad()
            model = model.train()
            pred, trans, trans_feat = model(pcld)
            loss = F.nll_loss(pred, label)
            if feat_trans:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()

            if i % 10 == 0:
                j, data = next(enumerate(test))
                pcld, label = data
                pcld = pcld.transpose(2, 1)
                pcld, label = pcld.to(dev), label.to(dev)
                model = model.eval()
                pred, _, _ = model(pcld)
                loss = F.nll_loss(pred, label)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(label.data).cpu().sum()

        # torch.save(model.state_dict(), "%s/cls_model_%d.pth" % (opt.outf, epoch))


def eval_model(model, test):
    total_correct = 0
    total_testset = 0
    total_classes = {}
    correct_classes = {}
    for data in tqdm(test):
        pcld, label = data
        pcld = pcld.transpose(2, 1)
        pcld, label = pcld.to(dev), label.to(dev)
        model = model.eval()
        pred, _, _ = model(pcld)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label.data)
        for c, l in zip(correct, label):
            c = c.item()
            l = l.item()
            if l not in total_classes:
                total_classes[l] = 0
            if l not in correct_classes:
                correct_classes[l] = 0
            correct_classes[l] += c
            total_classes[l] += 1
        total_correct += correct.cpu().sum().item()
        total_testset += pcld.size()[0]
    for cls in correct_classes.keys():
        print(
            f"{classes[int(cls)]}: {correct_classes[cls]}/{total_classes[cls]}={correct_classes[cls]/total_classes[cls]}"
        )
    return total_correct / float(total_testset)
