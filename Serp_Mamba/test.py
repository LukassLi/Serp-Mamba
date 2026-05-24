import argparse
import os
import shutil
import logging
import h5py
import matplotlib.pyplot as plt
import numpy as np
np.bool = bool
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
# from distance_metrics_fast import hd95_fast, asd_fast, nsd
from PIL import Image
from networks.net_factory import net_factory
from dataloaders.dataset import BaseDataSets
from dataloaders.dataset_registry import load_dataset_config, ConfigDataSets
from torch.utils.data import DataLoader
import torch.nn as nn
from U_Mamba_main.umamba.nnunetv2.nets.SerpMamba import SerpMamba
import torch
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/lishh237/Serp-Mamba/Serp_Mamba/PRIME-FP20_DataPort/PRIME-FP20-TEST/val1_test", help="Name of Experiment")    #最后别加“/”
parser.add_argument("--exp", type=str, default="SerpMamba", help="experiment_name")
parser.add_argument('--model', type=str,
                    default='unet', help='data_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--checkpoint', type=str,  default="best",
                    help='last or best')
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch_size per gpu")
parser.add_argument("--patch_size", type=list,
                    default=[1024, 1024], help="patch size of network input")
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
parser.add_argument("--dataset", type=str, default="prime_fp20",
                    help="Dataset config name or path to YAML file")
parser.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Directory containing .pth checkpoint files")
parser.add_argument("--split", type=str, default="test",
                    help="Dataset split to evaluate: test, val, or train")
parser.add_argument("--save_dir", type=str, default=None,
                    help="Directory to save per-image predictions and metrics (default: experiments/<dataset>/test_results)")
args = parser.parse_args()

# 加载数据集配置
cfg_path = args.dataset
if not cfg_path.endswith('.yaml'):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', cfg_path + '.yaml')
dataset_cfg = load_dataset_config(cfg_path)
if args.root_path != parser.get_default("root_path"):
    dataset_cfg["root_dir"] = args.root_path

unet_config = {"UNet_base_num_features": 32,
                "n_conv_per_stage_encoder": [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2
                ],
                "n_conv_per_stage_decoder": [
                    2,
                    2,
                    2,
                    2,
                    2
                ],
                "num_pool_per_axis": [
                    5,
                    5,
                    5
                ],
                "pool_op_kernel_sizes": [
                    [
                        1,
                        1,
                    ],
                    [
                        2,
                        2,
                    ],
                    [
                        2,
                        2,
                    ],
                    [
                        2,
                        2,
                    ],
                    [
                        2,
                        2,
                    ],
                    [
                        2,
                        2,
                    ]
                ],
                "conv_kernel_sizes": [
                    [
                        3,
                        3,
                    ],
                    [
                        3,
                        3,
                    ],
                    [
                        3,
                        3,
                    ],
                    [
                        3,
                        3,
                    ],
                    [
                        3,
                        3,
                    ],
                    [
                        3,
                        3,
                    ]
                ],
                "unet_max_num_features": 320,
}
num_stages = len(unet_config["conv_kernel_sizes"])
conv_op = nn.Conv2d

# 与train.py中的other_kwargs保持一致，避免参数不一致导致错误，保证模型结构与训练时完全一致，checkpoint 加载不会再出现 Unexpected key(s) 的错误
other_kwargs = {
'conv_bias': True,
'norm_op': nn.InstanceNorm2d,
'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
'dropout_op': None, 'dropout_op_kwargs': None,
'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True}
}


def calculate_bm(pred, gt):
    # 计算TP, FP, TN, FN
    TP = ((pred == 1) & (gt == 1)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    TN = ((pred == 0) & (gt == 0)).sum()
    FN = ((pred == 0) & (gt == 1)).sum()
    
    # 计算真正率（TPR）和真负率（TNR）
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # 计算BM
    BM = TPR + TNR - 1
    return BM

def calculate_mcc(pred, gt):
    
    # 计算TP, FP, TN, FN
    TP = ((pred == 1) & (gt == 1)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    TN = ((pred == 0) & (gt == 0)).sum()
    FN = ((pred == 0) & (gt == 1)).sum()
    
    # 计算MCC
    mcc_denominator = np.sqrt(float(TP+FP) * float(TP+FN) * float(TN+FP) * float(TN+FN))
    MCC = ((TP * TN) - (FP * FN)) / mcc_denominator if mcc_denominator > 0 else 0
    return MCC

def calculate_metric_percase1(pred, gt):
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0
    
def calculate_metric_percase2(pred, gt):
    if pred.sum() > 0:
        iou = metric.binary.jc(pred,gt)
        return iou
    else:
        return 0

#2d
def test_single_volume_fast(case, image, label, net, classes, patch_size=[1024, 1024], save_path=None):
    # 将输入图像和标签转换为numpy数组
    image, label = image.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy()

    # 初始化预测数组
    prediction = np.zeros_like(label)

    # 缩放图像以匹配网络输入尺寸
    x, y = image.shape
    zoomed_image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)

    # 将处理后的图像转换为适合网络输入的格式
    input = torch.from_numpy(zoomed_image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        softmax_probs = torch.softmax(net(input), dim=1)
        uncertainty = -1.0 * torch.sum(softmax_probs * torch.log(softmax_probs + 1e-6), dim=1, keepdim=True)
        # print(softmax_probs.shape)
        out = torch.argmax(softmax_probs, dim=1)
        out = out.cpu().detach().numpy().squeeze()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
        if save_path is not None:
            # 按原始文件名保存预测掩码（白色=血管，黑色=背景）
            pred_img = Image.fromarray(prediction.astype(np.uint8) * 255)
            pred_img.save(save_path)
    metric_list1 = []
    metric_list2 = []
    metric_list3 = []
    metric_list4 = []
    for i in range(1, classes):

        metric_list1.append(calculate_metric_percase1(prediction == 1, label == 1))
        metric_list2.append(calculate_metric_percase2(prediction == 1, label == 1))
        metric_list3.append(calculate_mcc(prediction == 1, label == 1))
        metric_list4.append(calculate_bm(prediction == 1, label == 1))

    return metric_list1,metric_list2,metric_list3,metric_list4


def Inference(FLAGS):
    split = FLAGS.split
    db_test = ConfigDataSets(config=dataset_cfg, split=split)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)
    folder_path = FLAGS.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiments", dataset_cfg["name"])

    # 输出目录：保存预测掩码和指标
    save_dir = FLAGS.save_dir or os.path.join(folder_path, "test_results")
    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    files = os.listdir(folder_path)
    pth_files = [file for file in files if file.endswith(".pth")]
    sorted_files = sorted(pth_files)

    patch_size = dataset_cfg.get("patch_size", FLAGS.patch_size)

    with open(os.path.join(save_dir, 'output.txt'), 'w') as file:
        for file1 in sorted_files:
            print(os.path.join(folder_path, file1))

            snapshot_path = os.path.join(folder_path, file1)

            net = SerpMamba(input_channels=dataset_cfg.get("input_channels", 1), n_stages=len(unet_config["conv_kernel_sizes"]),features_per_stage=[min(unet_config["UNet_base_num_features"] * 2 ** i,
                            unet_config["unet_max_num_features"]) for i in range(num_stages)],conv_op=conv_op,kernel_sizes=unet_config["conv_kernel_sizes"],
                            strides=unet_config["pool_op_kernel_sizes"],n_conv_per_stage=unet_config["n_conv_per_stage_encoder"],n_conv_per_stage_decoder=unet_config['n_conv_per_stage_decoder'],
                        num_classes=FLAGS.num_classes,**other_kwargs)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = net.to(device)
            if FLAGS.checkpoint == "best":
                save_mode_path = snapshot_path
            else:
                save_mode_path = os.path.join(snapshot_path, 'model_iter_60000.pth')
            net.load_state_dict(torch.load(save_mode_path)["state_dict"])
            print("init weight from {}".format(save_mode_path))
            net.eval()

            metric_list = []
            metric_list2 = []
            metric_list3 = []
            metric_list4 = []
            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(testloader):
                    case_name = sampled_batch["name"][0] if isinstance(sampled_batch["name"], (list, tuple)) else sampled_batch["name"]
                    # 保存路径：以原始文件名（替换扩展名为 .png）命名
                    base_name = os.path.splitext(case_name)[0] + ".png"
                    save_path = os.path.join(pred_dir, base_name)

                    metric_i,metric_i2,metric_i3,metric_i4 = test_single_volume_fast(
                        sampled_batch["name"],
                        sampled_batch["image"],
                        sampled_batch["label"],
                        net,
                        classes=FLAGS.num_classes,
                        patch_size=patch_size,
                        save_path=save_path,
                    )
                    metric_list.append(np.array(metric_i))
                    metric_list2.append(np.array(metric_i2))
                    metric_list3.append(np.array(metric_i3))
                    metric_list4.append(np.array(metric_i4))

            performance = np.mean(metric_list)
            performance2 = np.mean(metric_list2)
            performance3 = np.mean(metric_list3)
            performance4 = np.mean(metric_list4)
            variance = np.std(metric_list)
            variance2 = np.std(metric_list2)
            variance3 = np.std(metric_list3)
            variance4 = np.std(metric_list4)
            print("iteration %s : mean_dice: %f" % (file1, performance))
            print("iteration %s : mean_iou: %f" % (file1, performance2))
            print("----------------------------\n")

            # 写入汇总指标
            file.write("model_name = " + snapshot_path + "\n")
            file.write("split = " + split + "\n")
            file.write("num_samples = %d\n" % len(db_test))
            file.write("Dice = mean-sd = " + str(performance) + "-" + str(variance) + "\n")
            file.write("Iou = mean-sd = " + str(performance2) + "-" + str(variance2) + "\n")
            file.write("MCC = mean-sd = " + str(performance3) + "-" + str(variance3) + "\n")
            file.write("BM = mean-sd = " + str(performance4) + "-" + str(variance4) + "\n")

            # 写入逐图指标
            file.write("\nper-image results:\n")
            for i in range(len(db_test)):
                case_name = db_test.sample_list[i]
                file.write("  %s: dice=%.6f, iou=%.6f, mcc=%.6f, bm=%.6f\n" % (
                    case_name, metric_list[i][0], metric_list2[i][0],
                    metric_list3[i][0], metric_list4[i][0]))
            file.write("\n")

    print("results saved to {}".format(save_dir))


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(FLAGS.root_path)
    print('Model = ',FLAGS.exp)
