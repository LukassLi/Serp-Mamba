"""
dataset_registry.py - 配置驱动的数据集加载，支持多数据集扩展。

通过 YAML 配置文件描述不同数据集的目录布局、文件名映射和图像属性，
由 ConfigDataSets 类统一加载，替代原有硬编码的 BaseDataSets。
"""

import os
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


def load_dataset_config(config_path: str) -> dict:
    """从 YAML 文件加载数据集配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    required_fields = ['name', 'root_dir', 'split_map', 'image_dir', 'label_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置文件 '{config_path}' 缺少必填字段 '{field}'")
    return config


def _transform_label_name(image_filename: str, transforms: list) -> str:
    """根据配置的变换规则，从图像文件名推导标签文件名。

    支持三种操作：
      - {"replace": [old, new]}   -> str.replace(old, new)
      - {"change_ext": ".png"}    -> 更改文件扩展名
      - {"append_suffix": "_xx"}  -> 在扩展名前插入后缀
    """
    name = image_filename
    for op in transforms:
        if "replace" in op:
            name = name.replace(op["replace"][0], op["replace"][1])
        elif "change_ext" in op:
            base, _ = os.path.splitext(name)
            name = base + op["change_ext"]
        elif "append_suffix" in op:
            base, ext = os.path.splitext(name)
            name = base + op["append_suffix"] + ext
    return name


class ConfigDataSets(Dataset):
    """配置驱动的数据集类，支持多种数据集格式。

    返回格式与 BaseDataSets 一致：{"image": tensor, "label": tensor, "name": str, "idx": int}
    """

    def __init__(self, config: dict, split: str = "train", transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        # 通过 split_map 将内部 split 名映射到实际目录名
        dir_split = config["split_map"].get(split, split)
        self.image_dir = os.path.join(
            config["root_dir"],
            config["image_dir"].replace("{split}", dir_split)
        )
        self.label_dir = os.path.join(
            config["root_dir"],
            config["label_dir"].replace("{split}", dir_split)
        )

        # TODO: 启用 sorted() 可使文件顺序跨平台确定化，但会改变与 BaseDataSets 的数据遍历顺序。
        #       回归验证通过后建议启用，提升可复现性。
        self.sample_list = os.listdir(self.image_dir)
        # 按配置的扩展名过滤
        if "image_ext" in config:
            ext = config["image_ext"]
            self.sample_list = [f for f in self.sample_list if f.endswith(ext)]

        print(f"[{config['name']}] {split} split: {len(self.sample_list)} samples "
              f"from {self.image_dir}")

    def __len__(self):
        return len(self.sample_list)

    def _load_image(self, path: str) -> np.ndarray:
        """按配置的颜色模式加载图像，若 input_channels=1 则转为灰度。"""
        img = Image.open(path)
        mode = self.config.get("image_mode", "L")
        img = img.convert(mode)
        # input_channels=1 时确保返回 2D 灰度数组，兼容 RandomGenerator
        if self.config.get("input_channels", 1) == 1 and img.mode != "L":
            img = img.convert("L")
        return np.array(img)

    def _load_label(self, image_filename: str) -> np.ndarray:
        """推导标签文件名并加载标签掩码。"""
        transforms = self.config.get("label_name_transform", [])
        label_name = _transform_label_name(image_filename, transforms)
        label_path = os.path.join(self.label_dir, label_name)
        label = np.array(Image.open(label_path))

        # RGB 标签转单通道
        if len(label.shape) == 3:
            mode = self.config.get("label_rgb_mode", "mean")
            if mode == "green":
                label = label[:, :, 1]
            elif mode == "first":
                label = label[:, :, 0]
            else:
                label = label.mean(axis=-1)

        # 二值化
        threshold = self.config.get("label_threshold", 0)
        label[label > threshold] = 1

        return label

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = self._load_image(os.path.join(self.image_dir, case))
        label = self._load_label(case)

        if self.split == "train" and self.transform is not None:
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["name"] = case
        sample["idx"] = idx
        return sample
