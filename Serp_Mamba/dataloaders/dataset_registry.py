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
    required_fields = ['name', 'root_dir', 'split_map', 'image_dir']
    if config.get("has_labels", True):
        required_fields.append('label_dir')
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
        self.has_labels = config.get("has_labels", True)

        # 通过 split_map 将内部 split 名映射到实际目录名
        dir_split = config["split_map"].get(split, split)
        # root_dir 支持字符串（所有 split 共用）或 dict（按 split 分别指定）
        root = config["root_dir"]
        if isinstance(root, dict):
            root = root[split]
        self.image_dir = os.path.join(
            root,
            config["image_dir"].replace("{split}", dir_split)
        )
        if self.has_labels:
            self.label_dir = os.path.join(
                root,
                config["label_dir"].replace("{split}", dir_split)
            )

        # TODO: 启用 sorted() 可使文件顺序跨平台确定化，但会改变与 BaseDataSets 的数据遍历顺序。
        #       回归验证通过后建议启用，提升可复现性。
        self.sample_list = os.listdir(self.image_dir)
        # 按配置的扩展名过滤
        if "image_ext" in config:
            ext = config["image_ext"]
            self.sample_list = [f for f in self.sample_list if f.endswith(ext)]

        # 支持 val_split_ratio：从训练目录按比例划出验证集
        # 当配置了 val_split_ratio 时，train/val 两个 split 都从同一目录读取，
        # 通过固定种子随机划分，保证可复现且类别近似均衡。
        # 当 test 与 train 共享同一目录时（如 CHASE_DB1 所有文件平铺在 raw/ 下），
        # test 也走 val_split_ratio 划分，避免加载训练数据导致数据泄漏。
        val_split_ratio = config.get("val_split_ratio", None)
        train_dir_key = config["split_map"].get("train", "train")
        test_dir_key = config["split_map"].get("test", "test")
        test_shares_dir = (train_dir_key == test_dir_key and
                           config["image_dir"].replace("{split}", train_dir_key) ==
                           config["image_dir"].replace("{split}", test_dir_key))
        apply_val_split = val_split_ratio is not None and (
            split in ("train", "val") or (split == "test" and test_shares_dir)
        )
        if apply_val_split:
            seed = config.get("val_split_seed", 1337)
            sorted_files = sorted(self.sample_list)
            rng = np.random.RandomState(seed)
            indices = list(range(len(sorted_files)))
            rng.shuffle(indices)
            n_val = round(len(indices) * val_split_ratio)
            val_indices = set(indices[:n_val])

            if split in ("val", "test"):
                self.sample_list = sorted([sorted_files[i] for i in val_indices])
            else:  # train
                self.sample_list = sorted([sorted_files[i] for i in set(indices) - val_indices])

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
        label_img = Image.open(label_path)
        if label_img.mode in ("RGBA", "LA", "PA"):
            label_img = label_img.convert("RGB")
        label = np.array(label_img)

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
        label = (label > threshold).astype(np.uint8)

        return label

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = self._load_image(os.path.join(self.image_dir, case))
        if self.has_labels:
            label = self._load_label(case)
        else:
            label = np.zeros(image.shape[:2], dtype=np.int16)

        if self.split == "train" and self.transform is not None:
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
        else:
            sample = {"image": image, "label": label.astype(np.int16)}

        sample["name"] = case
        sample["idx"] = idx
        return sample
