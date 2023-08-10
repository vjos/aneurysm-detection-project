import numpy as np


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


"""
rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
rotated_data = provider.rotate_point_cloud(batch_data)
rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
jittered_data = provider.shift_point_cloud(jittered_data)
jittered_data = provider.jitter_point_cloud(jittered_data)
rotated_data[:,:,0:3] = jittered_data
return provider.shuffle_points(rotated_data)
"""
