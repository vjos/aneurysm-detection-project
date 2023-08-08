import numpy as np
import os
import torch
from torch.utils.data import Dataset


""" Todo:
- train/valid/test sets
- figure out if the annotated data is a subset of the generated data (it should be)
- normalise to unit point if necessary
- implement augmentation: research appropriate pointcloud augmentation techniques
- aneurysm localisation via pyramid-style segmentation
"""


class IntrA(Dataset):
    def __init__(
        self,
        root,
        dataset="generated",
        npoints=2048,
        data_aug=True,
        exclude_seg=False,
        norm=False,
        fold=1,
        kfold_splits=None,
        test=False,
    ):
        root = os.path.expanduser(root)
        self.npoints = npoints
        self.paths, self.labels = [], []
        self.seg = not exclude_seg
        self.norm = norm

        if kfold_splits:
            if test:
                self.paths, self.labels = get_split_data(kfold_splits, root, fold)
            else:
                train_splits = [1, 2, 3, 4, 5]
                train_splits.remove(fold)
                for i in train_splits:
                    p, l = get_split_data(kfold_splits, root, i)
                    self.paths += p
                    self.labels += l
        else:
            if dataset == "generated":
                for i, cls in enumerate(["vessel", "aneurysm"]):
                    paths = get_paths(os.path.join(root, dataset, cls, "ad"))
                    self.paths += paths
                    self.labels += [i] * len(paths)
            elif dataset == "annotated":
                self.paths = get_paths(os.path.join(root, dataset, "ad"))
                self.labels = [1] * len(self.paths)
            elif dataset == "classification":
                for i, cls in enumerate(["vessel", "aneurysm"]):
                    paths = get_paths(os.path.join(root, dataset, cls, "ad"))
                    self.paths += paths
                    self.labels += [i] * len(paths)
                anno_paths = get_paths(os.path.join(root, "annotated", "ad"))
                self.paths += anno_paths
                self.labels += [1] * len(anno_paths)
            elif dataset == "complete":
                self.paths = get_paths(os.path.join(root, dataset))
                self.labels = [0] * len(self.paths)
            else:
                raise IOError(f"Unknown dataset type '{dataset}'.")

    def __getitem__(self, index):
        """Load pointcloud, sample/duplicate to correct npoints, then return with label."""
        pcld = load_pointcloud(self.paths[index], norm=self.norm, seg=self.seg)
        point_sample = np.random.choice(
            pcld.shape[0], self.npoints, replace=(pcld.shape[0] < self.npoints)
        )
        pcld = pcld[point_sample, :]

        return pcld, self.labels[index]

    def __len__(self):
        return len(self.paths)


def load_pointcloud(file, norm=False, seg=False):
    """Load an intra pointcloud as a tensor.
    .ad files: Each line represents a point: [(x,y,z), norm(x,y,z), seg_class]. Return [pcld, seg].
    .obj files: Lines prefixed with v represent the (x,y,z) coordinates for a point. Returns [pcld].
    """
    if (ext := os.path.splitext(file)[-1]) == ".ad":
        f = np.loadtxt(file)
        if not norm:
            f = f[:, [0, 1, 2, 6]]
        if not seg:
            f = f[:, :-1]
        return torch.from_numpy(f.astype(np.float32))
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


def get_paths(dir):
    """Returns list of full paths of all pointcloud files in a given directory."""
    return [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.splitext(f)[-1] in [".ad", ".obj"]
    ]


def get_split_data(splits_root, intra_root, fold):
    """Returns list of full paths of all pointcloud files in the given fold split."""
    paths = []
    labels = []

    # get aneurysms
    counter = 0
    with open(os.path.join(splits_root, f"ann_clsSplit_{fold-1}.txt")) as F:
        for line in F:
            paths.append(os.path.join(intra_root, line.strip()))
            counter += 1
    labels = [1] * counter

    # get vessels
    counter = 0
    with open(os.path.join(splits_root, f"negSplit_{fold-1}.txt")) as F:
        for line in F:
            paths.append(os.path.join(intra_root, line.strip()))
            counter += 1
    labels += [0] * counter

    return paths, labels


if __name__ == "__main__":
    dataset = IntrA("~/Documents/Datasets/IntrA")
