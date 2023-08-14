import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import sklearn.metrics as metrics

import Common.data_utils as d_utils
from Common import loss_utils

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
    ):
        classifier.to(dev)
        augmentor.to(dev)

        if torch.cuda.device_count() > 1:
            print("Detected", torch.cuda.device_count(), "GPUs")
            classifier = nn.DataParallel(classifier)
            augmentor = nn.DataParallel(augmentor)

        global_epoch = 0
        PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()

        for epoch in range(epochs):
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

                augLoss.backward(retain_graph=True)
                optimizer_a.step()

                optimizer_c.zero_grad()
                clsLoss = loss_utils.cls_loss(
                    pred_pc,
                    pred_aug,
                    label,
                )
                clsLoss.backward(retain_graph=True)
                optimizer_c.step()

            train_acc = self.eval_one_epoch(classifier.eval(), train)
            test_acc = self.eval_one_epoch(classifier.eval(), test)

            print("CLS Loss: %.2f" % clsLoss.data)
            print("AUG Loss: %.2f" % augLoss.data)

            print("Train Accuracy: %f" % train_acc)
            print("Test Accuracy: %f" % test_acc)

            if scheduler_c is not None:
                scheduler_c.step()
            if scheduler_a is not None:
                scheduler_a.step()

            global_epoch += 1

    def eval_one_epoch(self, model, loader):
        mean_correct = []
        test_pred = []
        test_true = []

        for data in loader:
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = model.eval()
            pred = classifier(points)[0]
            pred_choice = pred.data.max(1)[1]

            test_true.append(target.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)

        return test_acc
