import argparse
import os
import numpy as np
np.bool = bool
import torch
from scipy.ndimage import zoom
from PIL import Image
from dataloaders.dataset_registry import load_dataset_config, ConfigDataSets
from torch.utils.data import DataLoader
import torch.nn as nn
from U_Mamba_main.umamba.nnunetv2.nets.SerpMamba import SerpMamba
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/lishh237/Serp-Mamba/Serp_Mamba/PRIME-FP20_DataPort/PRIME-FP20-TEST/val1_test", help="Name of Experiment")    #最后别加"/"
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
parser.add_argument('--test_all', action='store_true',
                    help='Test all .pth checkpoints instead of only best model')
parser.add_argument("--save_dir", type=str, default=None,
                    help="Directory to save per-image predictions and metrics (default: experiments/<dataset>/test_results)")
parser.add_argument('--evaluate', action='store_true',
                    help='Run evaluation after inference using the configured evaluator')
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


def run_inference(image, net, patch_size):
    """纯推理：图像 → softmax 概率图 + 二值预测。

    Args:
        image: torch.Tensor (1, C, H, W)
        net: 模型
        patch_size: 网络输入尺寸

    Returns:
        prediction: numpy array (H, W)，二值预测 (0/1)
        prob_map: numpy array (H, W)，类别 1 的 softmax 概率
    """
    image_np = image.squeeze().cpu().detach().numpy()
    x, y = image_np.shape

    # 缩放图像以匹配网络输入尺寸
    zoomed_image = zoom(image_np, (patch_size[0] / x, patch_size[1] / y), order=0)
    input_tensor = torch.from_numpy(zoomed_image).unsqueeze(0).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        softmax_probs = torch.softmax(net(input_tensor), dim=1)
        out = torch.argmax(softmax_probs, dim=1).cpu().detach().numpy().squeeze()

    # 缩放回原始尺寸
    prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    prob_class1 = softmax_probs[0, 1].cpu().detach().numpy()
    prob_map = zoom(prob_class1, (x / patch_size[0], y / patch_size[1]), order=0)

    return prediction, prob_map


def Inference(FLAGS):
    split = FLAGS.split
    db_test = ConfigDataSets(config=dataset_cfg, split=split)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)
    folder_path = FLAGS.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiments", dataset_cfg["name"])

    # 输出目录：保存预测掩码和概率图
    save_dir = FLAGS.save_dir or os.path.join(folder_path, "test_results")
    pred_dir = os.path.join(save_dir, "predictions")
    prob_dir = os.path.join(save_dir, "probabilities")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)

    files = os.listdir(folder_path)
    pth_files = [file for file in files if file.endswith(".pth")]

    # 默认只测试 best model，--test_all 时遍历所有 checkpoint
    if not FLAGS.test_all:
        best_candidates = [f for f in pth_files if "best_model" in f]
        if best_candidates:
            sorted_files = sorted(best_candidates)
        else:
            print("Warning: no *_best_model.pth found, falling back to all .pth files")
            sorted_files = sorted(pth_files)
    else:
        sorted_files = sorted(pth_files)

    patch_size = dataset_cfg.get("patch_size", FLAGS.patch_size)

    for file1 in sorted_files:
        print(os.path.join(folder_path, file1))

        snapshot_path = os.path.join(folder_path, file1)

        net = SerpMamba(input_channels=dataset_cfg.get("input_channels", 1), n_stages=len(unet_config["conv_kernel_sizes"]),features_per_stage=[min(unet_config["UNet_base_num_features"] * 2 ** i,
                        unet_config["unet_max_num_features"]) for i in range(num_stages)],conv_op=conv_op,kernel_sizes=unet_config["conv_kernel_sizes"],
                        strides=unet_config["pool_op_kernel_sizes"],n_conv_per_stage=unet_config["n_conv_per_stage_encoder"],n_conv_per_stage_decoder=unet_config['n_conv_per_stage_decoder'],
                    num_classes=FLAGS.num_classes,**other_kwargs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        net.load_state_dict(torch.load(snapshot_path)["state_dict"])
        print("init weight from {}".format(snapshot_path))
        net.eval()

        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(testloader):
                case_name = sampled_batch["name"][0] if isinstance(sampled_batch["name"], (list, tuple)) else sampled_batch["name"]
                base_name = os.path.splitext(case_name)[0] + ".png"

                # 纯推理
                prediction, prob_map = run_inference(
                    sampled_batch["image"], net, patch_size
                )

                # 保存预测掩码（白色=血管，黑色=背景）
                pred_path = os.path.join(pred_dir, base_name)
                pred_img = Image.fromarray(prediction.astype(np.uint8) * 255)
                pred_img.save(pred_path)

                # 保存概率图（供后续测评用，AUC 等指标需要）
                prob_path = os.path.join(prob_dir, os.path.splitext(base_name)[0] + "_prob.npy")
                np.save(prob_path, prob_map)

        print("iteration %s : predictions saved" % file1)
        print("----------------------------\n")

    print("predictions saved to {}".format(save_dir))

    # 推理完成后，若指定 --evaluate 则自动运行测评
    if FLAGS.evaluate:
        from evaluate import run_evaluation
        print("\n========== Running evaluation ==========")
        run_evaluation(
            config=dataset_cfg,
            pred_dir=pred_dir,
            prob_dir=prob_dir,
            save_dir=save_dir,
            checkpoint_names=sorted_files,
            split=split,
        )


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
    print("Dataset:", dataset_cfg["name"])
    print("Model:", FLAGS.exp)
