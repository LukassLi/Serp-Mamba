import argparse
import logging
import os
import random
import sys
import numpy as np
np.bool = bool
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from U_Mamba_main.umamba.nnunetv2.nets.SerpMamba import SerpMamba
from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator
)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume, test_image_fast

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/lishh237/Serp-Mamba/Serp_Mamba/PRIME-FP20_DataPort/PRIME-FP20-after-VAL1/", help="Path of Experiment")
parser.add_argument("--exp", type=str, default="Serp_Mamba", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int,
                    default=60, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=2,
                    help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.0001,
                    help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list,
                    default=[1024, 1024], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=2,
                    help="output channel of network")
parser.add_argument("--load", default=False,
                    action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)

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

other_kwargs = {
'conv_bias': True,
'norm_op': nn.InstanceNorm2d,
'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
'dropout_op': None, 'dropout_op_kwargs': None,
'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True}
}


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    print("max_iterations:", max_iterations)

    def create_model(ema=False):
        model = SerpMamba(input_channels=1, n_stages=len(unet_config["conv_kernel_sizes"]),features_per_stage=[min(unet_config["UNet_base_num_features"] * 2 ** i,
                                unet_config["unet_max_num_features"]) for i in range(num_stages)],conv_op=conv_op,kernel_sizes=unet_config["conv_kernel_sizes"],
                                strides=unet_config["pool_op_kernel_sizes"],n_conv_per_stage=unet_config["n_conv_per_stage_encoder"],n_conv_per_stage_decoder=unet_config['n_conv_per_stage_decoder'],
                            num_classes=num_classes,**other_kwargs)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=RandomGenerator(args.patch_size))
    with torch.no_grad():
        db_val = BaseDataSets(base_dir=args.root_path, split="val")

    model = create_model()
    model.cuda()
    iter_num = 0
    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    ce_loss = CrossEntropyLoss()

    writer = SummaryWriter("/home/lishh237/Serp-Mamba/Serp_Mamba/tf-logs/")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0
    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=120)

    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = (
                sampled_batch["image"],
                sampled_batch["label"],
            )
            image_batch, label_batch = (
                image_batch.cuda(),
                label_batch.cuda(),
            )
            outputs = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            # print(outputs_soft.shape)
            loss = 0.5 * (ce_loss(outputs, label_batch.long(
            )) + losses.dice_loss(outputs_soft[:, 1, ...], label_batch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/model_loss", loss, iter_num)


            if iter_num > 0 and iter_num % 10 == 0:
                model.eval()
                metric_list = 0.0
                metric_list2 = 0.0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i,metric_i2 = test_image_fast(
                            sampled_batch["image"],
                            sampled_batch["label"],
                            model,
                            classes=num_classes,
                            patch_size=args.patch_size
                        )
                        metric_list += np.array(metric_i)
                        metric_list2 += np.array(metric_i2)
                    epoch_loss = 0
                    metric_list = metric_list / len(db_val)
                    metric_list2 = metric_list2 / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model_val_{}_dice/iou".format(class_i + 1),
                        metric_list[class_i],
                        iter_num,
                    )

                performance = np.mean(metric_list)
                performance2 =np.mean(metric_list2)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model.pth".format(args.model))
                    util.save_checkpoint(
                        epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(
                        epoch_num, model, optimizer, loss, save_best)

                logging.info(
                    "iteration %d : model_mean_dice: %f, loss: %f" % (
                        iter_num, performance, loss.item())
                )
                logging.info(
                    "iteration %d : model_mean_iou: %f" % (
                        iter_num, performance2)
                )
            model.train()

            if iter_num % 30 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "model_iter_" + str(iter_num) + ".pth")
                util.save_checkpoint(
                    epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/home/lishh237/Serp-Mamba/Serp_Mamba/{}_{}".format(
        args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
