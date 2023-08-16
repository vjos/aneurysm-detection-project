import numpy as np
import torch


def pcld_shuffle(batch):
    idx = np.arange(batch.shape[1])
    np.random.shuffle(idx)
    return batch[:, idx, :]


def pcld_dropout(batch, max_dropout=0.85):
    """Apply random point dropout (corruption) to the input pointcloud batch. Max dropout is the maximum ratio 0-max to dropout."""
    batch_size, npoints, channels = batch.shape

    for i in range(batch_size):
        ratio = np.random.random() * 0.85

        # generate random numbers for the number of points
        vals = np.random.random(npoints)

        # get the indices where the random value falls below the ratio
        arr = np.where(vals <= ratio)[0]

        # set the selected indices to the same value as point 0; dropping information whilst keeping the input size the same
        if len(arr) > 0:
            batch[i, arr, :] = batch[i, 0, :]

    return batch


def pcld_shift(batch, dist=0.1):
    """Randomly shift the pointclouds in the batch."""
    batch_size, npoints, channels = batch.shape

    # generate unique (x,y,z) shift for each pcld in the batch using uniform distribution
    shifts = np.random.uniform(-dist, dist, (batch_size, 3))

    # apply the shift to each respective pcld
    for i in range(batch_size):
        batch[i, :, :] += shifts[i, :]

    return batch


def pcld_scale(batch, scale_range=(0.8, 1.2)):
    """Apply random scaling over a uniform dist. to batch of pointclouds, within the given range."""
    batch_size, npoints, channels = batch.shape

    # generate unique scale factor for each pointcloud in the batch
    scales = np.random.uniform(scale_range[0], scale_range[1], batch_size)

    # apply scaling to each respective pointcloud
    for i in range(batch_size):
        batch[i, :, :] *= scales[i]

    return batch


def rotate_pcld(batch):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch.shape, dtype=np.float32)
    for k in range(batch.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def pcld_jitter(batch, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch
    return jittered_data


def apply_augmentations(
    batch,
    rotation=True,
    scaling=True,
    jittering=True,
    translation=True,
    dropout=True,
    shuffle=True,
):
    if rotation:
        batch = rotate_pcld(batch.data.numpy())
    if scaling:
        batch = pcld_scale(batch)
    if jittering:
        batch = pcld_jitter(batch)
    if translation:
        batch = pcld_shift(batch)
    if dropout:
        batch = pcld_dropout(batch)
    if shuffle:
        batch = pcld_shuffle(batch)

    return torch.Tensor(batch)
