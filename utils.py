from __future__ import print_function
import argparse
import boto3
import numpy as np
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

session = boto3.session.Session(profile_name="masters-root")
s3 = session.client("s3")


def train_model(
    model,
    train,
    test,
    epochs=5,
    eval_epoch=10,
    feat_trans=False,
    model_name="model",
    snapshot_path="./snapshots",
):
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.to(dev)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    hist = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}", end="\r")
        scheduler.step()
        for i, (pcld, label) in enumerate(train):
            pcld = pcld.transpose(2, 1)
            pcld, label = pcld.to(dev), label.to(dev)
            optimizer.zero_grad()
            model = model.train()
            pred = model(pcld)[0]
            loss = F.nll_loss(pred, label)

            # if feat_trans:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001

            loss.backward()
            optimizer.step()

        # evaluate the model
        hist["train_acc"].append(ta := eval_model(model, train))
        hist["valid_acc"].append(va := eval_model(model, test))
        model.train()

        # print metrics and save checkpoint
        if epoch % eval_epoch == 0:
            print(f"Epoch: {epoch}, Train Accuracy: {ta}, Validaiton Accuracy: {va}")

            checkpoint = f"{model_name}_{epoch}.ckpt"
            checkpoint_path = os.path.join(snapshot_path, checkpoint)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "history": hist,
                    "epochs": epoch,
                },
                checkpoint_path,
            )
            s3.upload_file(checkpoint_path, "masters-models-bucket", checkpoint)


def eval_model(model, test, verbose=False):
    total_correct = 0
    total_testset = 0
    total_classes = {}
    correct_classes = {}
    for data in tqdm(test, disable=(not verbose)):
        pcld, label = data
        pcld = pcld.transpose(2, 1)
        pcld, label = pcld.to(dev), label.to(dev)
        model.eval()
        pred = model(pcld)[0]
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

    if verbose:
        print(correct_classes)
        print("Class Accuracies:")
        for cls in correct_classes.keys():
            print(
                f"{classes[int(cls)]}: {correct_classes[cls]}/{total_classes[cls]}={correct_classes[cls]/total_classes[cls]}"
            )

    return total_correct / float(total_testset)
