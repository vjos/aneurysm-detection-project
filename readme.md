## Instructions
To run all the models in this repo successfully, a CUDA GPU is required. Set up the environment, either through the dockerfile or by manually installing the requirements and pointnet2_ops_lib/ with pip.

The models were evaluated using the 5-fold cross validation splits from the IntrA dataset (data not included in this repo).
The `models.ipynb` notebook contains example code required to train the PointMLP and CurveNet model on a single fold of the splits.

## Acknowledgements
The `file_splits` are from the [IntrA dataset](https://github.com/intra3d2019/IntrA)

```
@InProceedings{yang2020intra,
  author = {Yang, Xi and Xia, Ding and Kin, Taichi and Igarashi, Takeo},
  title = {IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

`loss.py` and `pointmlp.py` contain code from [PointMLP](https://github.com/ma-xu/pointMLP-pytorch)

```
@article{ma2022rethinking,
    title={Rethinking network design and local geometry in point cloud: A simple residual MLP framework},
    author={Ma, Xu and Qin, Can and You, Haoxuan and Ran, Haoxi and Fu, Yun},
    journal={arXiv preprint arXiv:2202.07123},
    year={2022}
}
```

`curvenet.py`, `curvenet_util.py` and `walk.py` contain code from [CurveNet](https://github.com/tiangexiang/CurveNet)

```
@InProceedings{Xiang_2021_ICCV,
    author    = {Xiang, Tiange and Zhang, Chaoyi and Song, Yang and Yu, Jianhui and Cai, Weidong},
    title     = {Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {915-924}
}
```

`pointnet2_ops_lib/` is taken from [this PointNet++ implementation](https://github.com/erikwijmans/Pointnet2_PyTorch)

```
@article{pytorchpointnet++,
Author = {Erik Wijmans},
Title = {Pointnet++ Pytorch},
Journal = {https://github.com/erikwijmans/Pointnet2_PyTorch},
Year = {2018}
}

@inproceedings{qi2017pointnet++,
title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
booktitle={Advances in Neural Information Processing Systems},
pages={5099--5108},
year={2017}
}
```
