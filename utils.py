from __future__ import print_function
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from models.pointnet import feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import mlflow

classes = ["vessel", "aneurysm"]

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_model(
    model,
    train,
    test,
    epochs=5,
    checkpoint_epoch=10,
    feat_trans=False,
    model_name="model",
    snapshot_path="./snapshots",
    norm=False,
):
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.to(dev)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    hist = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    if exp := mlflow.get_experiment_by_name(model_name):
        id = exp.experiment_id
    else:
        id = mlflow.create_experiment(model_name)

    with mlflow.start_run(experiment_id=id):
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}", end="\r")
            scheduler.step()
            for i, (pcld, label) in enumerate(train):
                pcld = pcld.transpose(2, 1)
                pcld, label = pcld.to(dev), label.to(dev)
                optimizer.zero_grad()
                model = model.train()
                if norm:
                    pred = model(pcld[:, :3, :], pcld[:, 3:, :])
                else:
                    pred = model(pcld)[0]
                loss = F.nll_loss(pred, label)

                # if feat_trans:
                #     loss += feature_transform_regularizer(trans_feat) * 0.001

                loss.backward()
                optimizer.step()

            # evaluate the model
            t = eval_model(model, train, norm=norm)
            v = eval_model(model, test, norm=norm)
            hist["train_acc"].append(t[0])
            hist["valid_acc"].append(v[0])
            mlflow.log_metric("train_acc", t[0], step=epoch)
            mlflow.log_metric("valid_acc", v[0], step=epoch)
            mlflow.log_metric("valid_vessel_recall", v[1][0] / v[2][0], step=epoch)
            mlflow.log_metric("valid_aneurysm_recall", v[1][1] / v[2][1], step=epoch)
            model.train()

            # print metrics and save checkpoint
            if epoch % checkpoint_epoch == 0:
                print(
                    f"Epoch: {epoch}, Train Accuracy: {t[0]}, Validaiton Accuracy: {v[0]}"
                )

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
                mlflow.log_artifact(checkpoint_path, "checkpoints")


def eval_model(model, test, training_eval=False, verbose=False, norm=False):
    total_correct = 0
    total_testset = 0
    total_classes = {}
    correct_classes = {}
    for data in tqdm(test, disable=(not verbose)):
        pcld, label = data
        pcld = pcld.transpose(2, 1)
        pcld, label = pcld.to(dev), label.to(dev)
        model.eval()
        if norm:
            pred = model(pcld[:, :3, :], pcld[:, 3:, :])
        else:
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

    return total_correct / float(total_testset), correct_classes, total_classes
