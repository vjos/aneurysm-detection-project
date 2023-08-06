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
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

classes = ["vessel", "aneurysm"]

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_step(model, scheduler, optimizer, data, norm=False):
    """Train the model once on the given dataset."""
    scheduler.step()
    for i, (pcld, label) in enumerate(data):
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


def save_checkpoint(model, snapshot_path, filename, checkpoint_dict={}, log=False):
    """Save a checkpoint to given dir and upload to mlflow if desired."""
    checkpoint_path = os.path.join(snapshot_path, filename)
    torch.save(
        {"state_dict": model.state_dict()} | checkpoint_dict,
        checkpoint_path,
    )
    if log:
        mlflow.log_artifact(checkpoint_path, "checkpoints")


def train_setup(model_name, snapshot_path):
    """Function called before all training methods. Currently returns the MLflow experiment ID."""
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if exp := mlflow.get_experiment_by_name(model_name):
        id = exp.experiment_id
    else:
        id = mlflow.create_experiment(model_name)

    return id


def train_model(
    model,
    train,
    test,
    epochs=100,
    checkpoint_epoch=10,
    model_name="model",
    snapshot_path="./snapshots",
    norm=False,
):
    model.to(dev)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    exp_id = train_setup(model_name, snapshot_path)

    with mlflow.start_run(experiment_id=exp_id):
        for epoch in range(1, epochs + 1):
            train_step(model, scheduler, optimizer, train, norm=norm)

            train_metrics = eval_model_classification(
                model, train, norm=norm, prefix="train_"
            )

            test_metrics = eval_model_classification(
                model, test, norm=norm, prefix="test_"
            )

            mlflow.log_metrics(train_metrics | test_metrics, step=epoch)

            # save model checkpoint at the given epochs
            if checkpoint_epoch and epoch % checkpoint_epoch == 0:
                save_checkpoint(
                    model,
                    snapshot_path,
                    f"{model_name}_{epoch}.ckpt",
                    checkpoint_dict={"epoch": epoch},
                    log=True,
                )


def train_kfold(
    model_class,
    model_kwargs,
    dataset,
    folds=5,
    epochs=100,
    batch_size=8,
    num_workers=0,
    norm=False,
    checkpoint_epoch=None,
    model_name="Model",
    snapshot_path="./snapshots",
):
    exp_id = train_setup(model_name, snapshot_path)

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"{folds}fold_{model_name}_{epochs}e"
    ):
        for fold, (train_ids, test_ids) in enumerate(kf.split(dataset), start=1):
            print(f"F{fold}")

            # instantiate a new instance of the model
            model = model_class(**model_kwargs)
            model.to(dev)
            optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            # get the split dataloaders for this fold
            train_srs = SubsetRandomSampler(train_ids)
            test_srs = SubsetRandomSampler(test_ids)
            train_dl = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=train_srs,
            )
            test_dl = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=test_srs,
            )

            for epoch in range(1, epochs + 1):
                print(f"Epoch: {epoch}", end="\r")
                train_step(model, scheduler, optimizer, train_dl, norm=norm)

                train_metrics = eval_model_classification(
                    model, train_dl, norm=norm, prefix=f"f{fold}_train_"
                )

                test_metrics = eval_model_classification(
                    model, test_dl, norm=norm, prefix=f"f{fold}_test_"
                )

                mlflow.log_metrics(train_metrics | test_metrics, step=epoch)
                if checkpoint_epoch and epoch % checkpoint_epoch == 0:
                    save_checkpoint(
                        model,
                        snapshot_path,
                        f"{model_name}_f{fold}{epoch}.ckpt",
                        checkpoint_dict={"epoch": epoch},
                        log=True,
                    )


def run_model(model, pcld, norm=False):
    """Returns the predictions generated from the model on the given batch."""
    pcld = pcld.transpose(2, 1)
    pcld = pcld.to(dev)

    if norm:
        pred = model(pcld[:, :3, :], pcld[:, 3:, :])
    else:
        pred = model(pcld)[0]

    return pred.data.max(1)[1]


def eval_model_classification(model, data, prefix="", norm=False):
    """Returns dictionary of appropriate metrics calculated by running the model on labelled data."""
    TP = N = 0
    total_classes, correct_classes = {}, {}
    model.eval()

    for pcld, label in data:
        label = label.to(dev)

        # generate predictions on batch and check prediction equality to groundtruths
        correct = run_model(model, pcld, norm=norm).eq(label.data)

        # sum TPs and total occurences for each class in the batch
        for c, l in zip(correct, label):
            c, l = c.item(), l.item()

            # add class labels to the dicts
            if l not in total_classes:
                total_classes[l] = 0
            if l not in correct_classes:
                correct_classes[l] = 0

            correct_classes[l] += c
            total_classes[l] += 1

        TP += correct.cpu().sum().item()
        N += pcld.size()[0]

    model.train()

    # return dictionary of metrics with prefix prepended to each
    return {
        f"{prefix}{m}": val
        for (m, val) in {
            "acc": TP / float(N),
            "vessel_recall": correct_classes[0] / total_classes[0],
            "aneurysm_recall": correct_classes[1] / total_classes[1],
        }.items()
    }
