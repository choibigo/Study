from .cityscapes import CitySegmentation
from .pascal_voc import VOCSegmentation

datasets = {
    'citys': CitySegmentation,
    'pascal_voc' : VOCSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
