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

classes = ["vessel", "aneurysm"]


dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_step(
    model,
    scheduler,
    optimizer,
    data,
    loss_fn=F.nll_loss,
    aug=None,
    trans_loss=False,
):
    """Train the model once on the given dataset."""
    model = model.train()

    for batch, label in data:
        optimizer.zero_grad()

        # apply data augmentation
        if aug:
            batch = aug(batch)

        # prepare data for model
        batch = batch.transpose(2, 1)
        batch, label = batch.to(dev), label.to(dev)

        # calculate model loss
        if trans_loss:
            pred, trans_feat = model(batch)
            loss = loss_fn(pred, label, trans_feat)
        else:
            pred = model(batch)[0]
            loss = loss_fn(pred, label)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    scheduler.step()


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
    loss_fn,
    aug,
    epochs=100,
    checkpoint_epoch=10,
    model_name="model",
    opt="adam",
    sched="step",
    lr=0.001,
    min_lr=0.001,
    momentum=0.9,
    weight_decay=2e-4,
    snapshot_path="./snapshots",
    trans_loss=False,
):
    model.to(dev)

    if opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr * 100,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
        )

    if sched == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=min_lr, last_epoch=-1
        )

    exp_id = train_setup(model_name, snapshot_path)
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_params({x: str(y) for x, y in locals().items() if x != "model"})
        for epoch in range(1, epochs + 1):
            train_step(
                model,
                scheduler,
                optimizer,
                train,
                loss_fn=loss_fn,
                aug=aug,
                trans_loss=trans_loss,
            )

            train_metrics = eval_model_classification(model, train, prefix="train_")

            test_metrics = eval_model_classification(model, test, prefix="test_")

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
    loss_fn,
    aug,
    opt="adam",
    sched="step",
    lr=0.001,
    momentum=0.9,
    epochs=100,
    batch_size=8,
    num_workers=0,
    norm=False,
    checkpoint_epoch=None,
    model_name="Model",
    intra_root="./data",
    npoints=1024,
    exclude_seg=True,
    snapshot_path="./snapshots",
    splits="./fileSplits",
    trans_loss=False,
):
    exp_id = train_setup(model_name, snapshot_path)
    cv_metrics = {}  # used to track cross-validation metrics

    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"5fold_{model_name}_{epochs}e_bs{batch_size}"
    ):
        mlflow.log_params({x: str(y) for x, y in locals().items() if x != "model"})
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
            if opt == "sgd":
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=lr * 100,
                    momentum=momentum,
                    weight_decay=1e-4,
                )
            else:
                optimizer = optim.Adam(
                    model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
                )

            if sched == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=20, gamma=0.7
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, epochs, eta_min=lr, start_epoch=-1
                )

            for epoch in range(1, epochs + 1):
                print(f"Epoch: {epoch}", end="\r")
                train_step(
                    model,
                    scheduler,
                    optimizer,
                    train_dl,
                    loss_fn=loss_fn,
                    aug=aug,
                    trans_loss=trans_loss,
                )

                train_metrics = eval_model_classification(
                    model, train_dl, prefix=f"f{fold}_train_"
                )

                test_metrics = eval_model_classification(
                    model, test_dl, prefix=f"f{fold}_test_"
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


def run_model(model, batch):
    """Returns the predictions generated from the model on the given batch."""
    batch = batch.transpose(2, 1)
    batch = batch.to(dev)
    pred = model(batch)[0]
    return pred.data.max(1)[1]


def eval_model_classification(model, data, prefix=""):
    """Returns dictionary of appropriate metrics calculated by running the model on labelled data."""

    def class_dict(n):
        return {i: 0 for i in range(n)}

    total_classes, tp_classes, fp_classes = class_dict(2), class_dict(2), class_dict(2)
    model.eval()

    for pcld, label in data:
        label = label.to(dev)

        # generate predictions on batch and check prediction equality to groundtruths
        pred = run_model(model, pcld)

        # sum TPs and total occurences for each class in the batch
        for p, l in zip(pred, label):
            p, l = p.item(), l.item()

            total_classes[l] += 1
            if p == l:
                tp_classes[l] += 1
            else:
                fp_classes[p] += 1

    model.train()

    ves_rec = tp_classes[0] / total_classes[0] if total_classes[0] else 0
    an_rec = tp_classes[1] / total_classes[1] if total_classes[1] else 0
    avg_rec = (ves_rec + an_rec) / 2
    ves_preds = tp_classes[0] + fp_classes[0]
    ves_pres = tp_classes[0] / ves_preds if ves_preds else 0
    an_preds = tp_classes[1] + fp_classes[1]
    an_pres = tp_classes[1] / an_preds if an_preds else 0
    avg_pres = (ves_pres + an_pres) / 2

    # return dictionary of metrics with prefix prepended to each
    return {
        f"{prefix}{m}": val
        for (m, val) in {
            "vessel_recall": ves_rec,
            "aneurysm_recall": an_rec,
            "avg_recall": avg_rec,
            "vessel_precision": ves_pres,
            "aneurysm_precision": an_pres,
            "avg_precision": avg_pres,
            "f1": (2 * avg_rec * avg_pres) / (avg_rec + avg_pres),
        }.items()
    }
