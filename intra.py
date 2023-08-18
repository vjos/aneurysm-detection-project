import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random


class IntrA(Dataset):
    def __init__(
        self,
        root,
        dataset="generated",
        npoints=2048,
        exclude_seg=False,
        norm=False,
        fold=1,
        kfold_splits=None,
        test=False,
        norm_only=False,
        oversample=False,
    ):
        root = os.path.expanduser(root)
        self.npoints = npoints
        self.paths, self.labels = [], []
        self.seg = not exclude_seg
        self.norm = norm
        self.norm_only = norm_only

        # decide between predefined splits or random dataset (kfold_splits only supported for classification tasks)
        if kfold_splits:
            if test:
                # load the test data for this fold
                self.paths, self.labels = get_split_data(kfold_splits, root, fold)
            else:
                # load the training data for this fold
                train_splits = [1, 2, 3, 4, 5]
                train_splits.remove(fold)
                for i in train_splits:
                    p, l = get_split_data(kfold_splits, root, i)
                    self.paths += p
                    self.labels += l
        else:
            # load only the generated data
            if dataset == "generated":
                for i, cls in enumerate(["vessel", "aneurysm"]):
                    paths = get_paths(os.path.join(root, dataset, cls, "ad"))
                    self.paths += paths
                    self.labels += [i] * len(paths)
            # load only annotated data (typically for segmentation)
            elif dataset == "annotated":
                self.paths = get_paths(os.path.join(root, dataset, "ad"))
                self.labels = [1] * len(self.paths)
            # load data for classification tasks
            elif dataset == "classification":
                for i, cls in enumerate(["vessel", "aneurysm"]):
                    paths = get_paths(os.path.join(root, "generated", cls, "ad"))
                    self.paths += paths
                    self.labels += [i] * len(paths)
                anno_paths = get_paths(os.path.join(root, "annotated", "ad"))
                self.paths += anno_paths
                self.labels += [1] * len(anno_paths)
            # load complete brain vessels data
            elif dataset == "complete":
                self.paths = get_paths(os.path.join(root, dataset))
                self.labels = [0] * len(self.paths)
            else:
                raise IOError(f"Unknown dataset type '{dataset}'.")

        label_indices = {}
        if oversample:
            # get the indices for each class, separated by label
            for idx, l in enumerate(self.labels):
                if l in label_indices:
                    label_indices[l].append(idx)
                else:
                    label_indices[l] = [idx]

            # if a class is under-represented, randomly oversample to match the size of the largest class
            most_common = max([len(x) for x in label_indices.values()])
            for l in label_indices:
                if (diff := most_common - len(label_indices[l])) > 0:
                    self.paths += [
                        self.paths[idx]
                        for idx in random.choices(label_indices[l], k=diff)
                    ]
                    self.labels += [l] * diff

    def __getitem__(self, index):
        """Load pointcloud, sample/duplicate to correct npoints, then return with label."""
        pcld = load_pointcloud(
            self.paths[index], norm=self.norm, norm_only=self.norm_only, seg=self.seg
        )

        point_sample = np.random.choice(
            pcld.shape[0], self.npoints, replace=(pcld.shape[0] < self.npoints)
        )

        pcld = pcld[point_sample, :]

        return pcld, self.labels[index]

    def __len__(self):
        return len(self.paths)


def load_pointcloud(file, norm=False, norm_only=False, seg=False):
    """Load an intra pointcloud as a tensor.
    .ad files: Each line represents a point: [(x,y,z), norm(x,y,z), seg_class]. Return [pcld, seg].
    .obj files: Lines prefixed with v represent the (x,y,z) coordinates for a point. Returns [pcld].
    """
    if (ext := os.path.splitext(file)[-1]) == ".ad":
        f = np.loadtxt(file)
        if norm_only:
            f = f[:, [3, 4, 5, 6]]
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
    """Returns list of full paths and labels of all pointcloud files in the given fold split."""
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
