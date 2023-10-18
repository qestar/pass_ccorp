
import torchvision

from ccorp import datasets
from ccorp.datasets.transforms import build_transform


def build_dataset(cfg):
    args = cfg.copy()

    # build transform
    transform = build_transform(args.trans_dict)

    # build dataset
    ds_dict = args.ds_dict
    ds_name = ds_dict.pop('type')
    ds_dict['transform'] = transform
    if hasattr(torchvision.datasets, ds_name):
        ds = getattr(torchvision.datasets, ds_name)(**ds_dict)
    else:
        ds = datasets.__dict__[ds_name](**ds_dict)
    return ds


def build_dataset_ccrop(cfg, ds_dict, rcrop_dict):
    args = cfg.copy()

    # build transform
    transform_rcrop = build_transform(rcrop_dict)
    transform_ccrop = build_transform(cfg)

    # build dataset
    ds_dict = ds_dict
    ds_name = ds_dict.pop('type')
    ds_dict['transform_rcrop'] = transform_rcrop
    ds_dict['transform_ccrop'] = transform_ccrop
    ds = datasets.__dict__[ds_name](**ds_dict)
    return ds