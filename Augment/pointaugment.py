import torch
import torch.nn as nn
import torch.utils.data
import Common.data_utils as d_utils
from Common import loss_utils
from utils import eval_model_classification, train_setup
import mlflow

# Set random seed for reproducibility
# import random
# manualSeed = 999
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class PointAugmentation:
    def __init__(self):
        pass

    def train(
        self,
        classifier,
        augmentor,
        train,
        test,
        epochs,
        optimizer_a,
        optimizer_c,
        scheduler_a,
        scheduler_c,
        model_name="PointAug",
        snapshot_path="./snapshots",
    ):
        classifier.to(dev)
        augmentor.to(dev)

        if torch.cuda.device_count() > 1:
            print("Detected", torch.cuda.device_count(), "GPUs")
            classifier = nn.DataParallel(classifier)
            augmentor = nn.DataParallel(augmentor)

        global_epoch = 0
        PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()

        exp_id = train_setup(model_name, snapshot_path)
        with mlflow.start_run(experiment_id=exp_id):
            for epoch in range(epochs):
                total_aug_loss = total_cls_loss = 0
                for batch, label in train:
                    batch, label = batch.to(dev), label.to(dev)

                    batch = PointcloudScaleAndTranslate(batch)
                    batch = batch.transpose(2, 1).contiguous()

                    noise = 0.02 * torch.randn(batch.size()[0], 1024).to(dev)

                    classifier = classifier.train()
                    augmentor = augmentor.train()
                    optimizer_a.zero_grad()
                    aug_pc = augmentor(batch, noise)

                    pred_pc = classifier(batch)[0]
                    pred_aug = classifier(aug_pc)[0]

                    augLoss = loss_utils.aug_loss(pred_pc, pred_aug, label)
                    total_aug_loss += augLoss

                    augLoss.backward(retain_graph=True)
                    optimizer_a.step()

                    optimizer_c.zero_grad()
                    clsLoss = loss_utils.cls_loss(
                        pred_pc,
                        pred_aug,
                        label,
                    )
                    total_cls_loss += clsLoss
                    clsLoss.backward(retain_graph=True)
                    optimizer_c.step()

                mlflow.log_metric("aug_loss", total_aug_loss / len(train), step=epoch)
                mlflow.log_metric("cls_loss", total_cls_loss / len(train), step=epoch)

                # evaluate classifier
                train_metrics = eval_model_classification(
                    classifier, train, prefix="train_"
                )
                test_metrics = eval_model_classification(
                    classifier, test, prefix="test_"
                )

                mlflow.log_metrics(train_metrics | test_metrics, step=epoch)
                mlflow.log_metric("lr_a", scheduler_a.get_last_lr()[0], step=epoch)
                mlflow.log_metric("lr_c", scheduler_c.get_last_lr()[0], step=epoch)

                if scheduler_c is not None:
                    scheduler_c.step()
                if scheduler_a is not None:
                    scheduler_a.step()

                global_epoch += 1
