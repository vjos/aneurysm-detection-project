import torch
import torch.nn.functional as F


dev = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class smooth_cross_entropy_loss(torch.nn.Module):
    def __init__(self, smoothing=True):
        super(smooth_cross_entropy_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, gt):
        gt = gt.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, reduction="mean")

        return loss
