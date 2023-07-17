import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

""" Todo:
- figure out if the annotated data is a subset of the generated data (it should be)
- normalise to unit point if necessary
- implement npoints restriction with np.random.choice
- implement augmentation: research appropriate pointcloud augmentation techniques
- data exploration/visualisation notebook
"""


class IntrA(Dataset):
    def __init__(self, root, train=True, npoints=2048, data_aug=True, choice=0):
        self.classes = ["vessel", "aneurysm"]
        self.paths, self.labels = [], []

        root = os.path.expanduser(root)
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(root, "generated", cls, "ad")
            paths = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if os.path.splitext(f)[-1] == ".ad"
            ]
            self.paths += paths
            self.labels += [i] * len(paths)

    def __getitem__(self, index):
        return load_pointcloud(self.paths[index], self.labels[index])


def load_pointcloud(file):
    """Load an intra pointcloud as a tensor from a .ad file. Each line represents a point: [(x,y,z), norm(x,y,z), seg_class]."""
    f = np.loadtxt(file)
    return torch.from_numpy(f[:, :3].astype(np.float32))


if __name__ == "__main__":
    dataset = IntrA("~/Documents/Datasets/IntrA")
    tst = os.path.expanduser(
        "~/Documents/Datasets/IntrA/generated/aneurysm/ad/ArteryObjAN2-0_addon.ad"
    )
