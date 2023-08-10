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
from torch.utils.data import SubsetRandomSampler, DataLoader
from intra import IntrA
from augmentation import pcld_dropout, pcld_shift, pcld_scale
from models.pointnetcls import get_loss

classes = ["vessel", "aneurysm"]

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_step(model, scheduler, optimizer, data, pointconv=False, augment_data=True):
    """Train the model once on the given dataset."""
    scheduler.step()
    loss_fn = get_loss()
    for batch, label in data:
        optimizer.zero_grad()
        model = model.train()

        if augment_data:
            # random dropout
            batch = pcld_dropout(batch.data.numpy())

            # scale and shift (wouldn't affect norm)
            batch[:, :, 0:3] = pcld_shift(pcld_scale(batch[:, :, 0:3]))

            batch = torch.Tensor(batch)

        # prepare batch for model
        batch = batch.transpose(2, 1)
        batch, label = batch.to(dev), label.to(dev)

        if pointconv:
            # works for pointconv
            pred = model(batch[:, :3, :], batch[:, 3:, :])
        else:
            pred, trans_feat = model(batch)

        # loss = F.nll_loss(pred, label)
        loss = loss_fn(pred, label, trans_feat)
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
    pointconv=False,
    augment_data=True,
):
    model.to(dev)
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    exp_id = train_setup(model_name, snapshot_path)

    with mlflow.start_run(experiment_id=exp_id):
        for epoch in range(1, epochs + 1):
            train_step(
                model,
                scheduler,
                optimizer,
                train,
                pointconv=pointconv,
                augment_data=augment_data,
            )

            train_metrics = eval_model_classification(
                model, train, norm=norm, pointconv=pointconv, prefix="train_"
            )

            test_metrics = eval_model_classification(
                model, test, norm=norm, pointconv=pointconv, prefix="test_"
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


def train_kfold_intra(
    model_class,
    model_kwargs,
    epochs=100,
    batch_size=8,
    num_workers=0,
    norm=False,
    pointconv=False,
    checkpoint_epoch=None,
    model_name="Model",
    intra_root="./data",
    npoints=1024,
    exclude_seg=True,
    snapshot_path="./snapshots",
    splits="./fileSplits",
    augment_data=True,
):
    exp_id = train_setup(model_name, snapshot_path)
    cv_metrics = {}  # used to track cross-validation metrics

    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"5fold_{model_name}_{epochs}e_bs{batch_size}"
    ):
        for fold in [1, 2, 3, 4, 5]:
            print(f"\nF{fold}:")
            trn = IntrA(
                intra_root,
                npoints=npoints,
                exclude_seg=exclude_seg,
                norm=norm,
                fold=fold,
                kfold_splits=splits,
                test=False,
            )
            tst = IntrA(
                intra_root,
                npoints=npoints,
                exclude_seg=exclude_seg,
                norm=norm,
                fold=fold,
                kfold_splits=splits,
                test=True,
            )

            # get the split dataloaders for this fold
            train_dl = DataLoader(
                trn, batch_size=batch_size, num_workers=num_workers, shuffle=True
            )
            test_dl = DataLoader(
                tst, batch_size=batch_size, num_workers=num_workers, shuffle=False
            )

            # instantiate a new instance of the model
            model = model_class(**model_kwargs)
            model.to(dev)
            optimizer = optim.Adam(
                model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

            for epoch in range(1, epochs + 1):
                print(f"Epoch: {epoch}", end="\r")
                train_step(
                    model,
                    scheduler,
                    optimizer,
                    train_dl,
                    pointconv=pointconv,
                    augment_data=augment_data,
                )

                train_metrics = eval_model_classification(
                    model, train_dl, norm=norm, prefix=f"f{fold}_train_"
                )

                test_metrics = eval_model_classification(
                    model, test_dl, norm=norm, prefix=f"f{fold}_test_"
                )

                # log all epoch metrics to mlflow
                full_metrics = train_metrics | test_metrics
                mlflow.log_metrics(full_metrics, step=epoch)

                # sum metrics across all folds
                for k, v in full_metrics.items():
                    m = "_".join(k.split("_")[1:])  # remove fold number from metric
                    if m not in cv_metrics:
                        cv_metrics[m] = np.zeros(epochs)
                    cv_metrics[m][epoch - 1] += v

                if checkpoint_epoch and epoch % checkpoint_epoch == 0:
                    save_checkpoint(
                        model,
                        snapshot_path,
                        f"{model_name}_f{fold}{epoch}.ckpt",
                        checkpoint_dict={"epoch": epoch},
                        log=True,
                    )

        # log the average of each summed metric to mlflow
        for m in cv_metrics:
            for epoch, val in enumerate(cv_metrics[m]):
                mlflow.log_metric(m, val / 5, step=epoch + 1)


def run_model(model, pcld, pointconv=False):
    """Returns the predictions generated from the model on the given batch."""
    pcld = pcld.transpose(2, 1)
    pcld = pcld.to(dev)

    if pointconv:
        pred = model(pcld[:, :3, :], pcld[:, 3:, :])
    else:
        pred = model(pcld)[0]

    return pred.data.max(1)[1]


def eval_model_classification(model, data, prefix="", norm=False, pointconv=False):
    """Returns dictionary of appropriate metrics calculated by running the model on labelled data."""
    TP = N = 0
    total_classes, correct_classes = {}, {}
    model.eval()

    for pcld, label in data:
        label = label.to(dev)

        # generate predictions on batch and check prediction equality to groundtruths
        correct = run_model(model, pcld, pointconv=pointconv).eq(label.data)

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
