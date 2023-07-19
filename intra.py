import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


""" Todo:
- train/valid/test sets
- figure out if the annotated data is a subset of the generated data (it should be)
- normalise to unit point if necessary
- implement augmentation: research appropriate pointcloud augmentation techniques
- aneurysm localisation via pyramid-style segmentation
"""


class IntrA(Dataset):
    def __init__(self, root, dataset="generated", npoints=2048, data_aug=True):
        root = os.path.expanduser(root)
        self.npoints = npoints
        self.paths, self.labels = [], []

        if dataset == "generated":
            for i, cls in enumerate(["vessel", "aneurysm"]):
                paths = self.get_paths(os.path.join(root, dataset, cls, "ad"))
                self.paths += paths
                self.labels += [i] * len(paths)
        elif dataset == "annotated":
            self.paths = self.get_paths(os.path.join(root, dataset, "ad"))
            self.labels = [1] * len(self.paths)
        elif dataset == "complete":
            self.paths = self.get_paths(os.path.join(root, dataset))
            self.labels = [0] * len(self.paths)
        else:
            raise IOError(f"Unknown dataset type '{dataset}'.")

    def __getitem__(self, index):
        """Load pointcloud, sample/duplicate to correct npoints, then return with label."""
        pcld = load_pointcloud(self.paths[index])
        point_sample = np.random.choice(
            pcld.shape[0], self.npoints, replace=(pcld.shape[0] < self.npoints)
        )
        pcld = pcld[point_sample, :]

        return pcld, self.labels[index]

    def get_paths(self, dir):
        """Returns list of full paths of all pointcloud files in a given directory."""
        return [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if os.path.splitext(f)[-1] in [".ad", ".obj"]
        ]


def load_pointcloud(file):
    """Load an intra pointcloud as a tensor.
    .ad files: Each line represents a point: [(x,y,z), norm(x,y,z), seg_class]. Return [pcld, seg].
    .obj files: Lines prefixed with v represent the (x,y,z) coordinates for a point. Returns [pcld].
    """
    if (ext := os.path.splitext(file)[-1]) == ".ad":
        f = np.loadtxt(file)
        return torch.from_numpy(f[:, [0, 1, 2, 6]].astype(np.float32))
    elif ext == ".obj":
        coords = []
        with open(file, "r") as F:
            for line in F:
                vals = line.strip().split(" ")
                if vals[0] == "v":
                    coords.append(vals[1:])
        return torch.tensor(
            np.hstack((np.array(coords), np.zeros((len(coords), 1)))).astype(np.float32)
        )
    else:
        raise IOError(f"File type {ext} is not supported.")


if __name__ == "__main__":
    dataset = IntrA("~/Documents/Datasets/IntrA")
