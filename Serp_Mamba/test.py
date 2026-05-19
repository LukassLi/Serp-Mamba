import argparse
import os
import shutil
import logging
import h5py
import matplotlib.pyplot as plt
import numpy as np
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
args = parser.parse_args()

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
def test_single_volume_fast(case,image, label, net, classes, patch_size=[1024, 1024]):
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
        # print("out,=",out)
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
        image = Image.fromarray(prediction.astype(np.uint8) * 255)
        image.save('/home/lishh237/Serp-Mamba/Serp_Mamba/output2_image.png')
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
    db_val = BaseDataSets(base_dir=FLAGS.root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    folder_path = "/home/lishh237/Serp-Mamba/Serp_Mamba/val1/"  # 加载权重文件
    files = os.listdir(folder_path)
    pth_files = [file for file in files if file.endswith(".pth")]
    sorted_files = sorted(pth_files)
    with open(folder_path+'output.txt', 'a') as file:
        for file1 in sorted_files:
            # if file1 == "unet_best_model.pth":
                print(folder_path+file1)

                snapshot_path = folder_path+file1

                net = SerpMamba(input_channels=1, n_stages=len(unet_config["conv_kernel_sizes"]),features_per_stage=[min(unet_config["UNet_base_num_features"] * 2 ** i,
                                unet_config["unet_max_num_features"]) for i in range(num_stages)],conv_op=conv_op,kernel_sizes=unet_config["conv_kernel_sizes"],
                                strides=unet_config["pool_op_kernel_sizes"],n_conv_per_stage=unet_config["n_conv_per_stage_encoder"],n_conv_per_stage_decoder=unet_config['n_conv_per_stage_decoder'],
                            num_classes=2)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                net = net.to(device)
                if FLAGS.checkpoint == "best":
                    save_mode_path = snapshot_path
                else:
                    save_mode_path = os.path.join(snapshot_path, 'model_iter_60000.pth')
                net.load_state_dict(torch.load(save_mode_path)["state_dict"])
                # net.load_state_dict(torch.load(save_mode_path, map_location=torch.device('cpu'))["state_dict"])
                print("init weight from {}".format(save_mode_path))
                net.eval()

                metric_list = []
                metric_list2 = []
                metric_list3 = []
                metric_list4 = []
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i,metric_i2,metric_i3,metric_i4 = test_single_volume_fast(
                            sampled_batch["name"], 
                            sampled_batch["image"],
                            sampled_batch["label"],
                            net,
                            classes=FLAGS.num_classes,
                        )
                        metric_list.append(np.array(metric_i))
                        metric_list2.append(np.array(metric_i2))
                        metric_list3.append(np.array(metric_i3))
                        metric_list4.append(np.array(metric_i4))
                performance = np.mean(metric_list)
                performance2 =np.mean(metric_list2)
                performance3 = np.mean(metric_list3)
                performance4 =np.mean(metric_list4)
                variance = np.std(metric_list)
                variance2 = np.std(metric_list2)
                variance3 = np.std(metric_list3)
                variance4 = np.std(metric_list4)
                print("iteration %s : mean_dice: %f" % (file1, performance))
                print("iteration %s : mean_iou: %f" % (file1, performance2))
                print("----------------------------\n")

                # 解释下面的代码：将模型名称、Dice、Iou、MCC、BM写入output.txt文件
                file.write("model_name = " + snapshot_path + "\n")

                file.write("Dice = mean-sd = " + str(performance) + "-" + str(variance) + "\n")
                file.write("Iou = mean-sd = " + str(performance2) + "-" + str(variance2) + "\n")
                file.write("MCC = mean-sd = " + str(performance3) + "-" + str(variance3) + "\n")
                file.write("BM = mean-sd = " + str(performance4) + "-" + str(variance4) + "\n")
                file.write("\n")


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(FLAGS.root_path)
    print('Model = ',FLAGS.exp)
