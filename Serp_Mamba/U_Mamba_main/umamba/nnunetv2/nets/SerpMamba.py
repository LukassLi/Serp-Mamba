import numpy as np
from typing import Union, Type, List, Tuple
from torch import nn
import einops
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
import torch
import torch.nn.functional as F
from PIL import Image

def map_features_back_to_input(input, point_coords, features):
    # Create a new tensor with the same shape as the input
    output = input.clone()

    # Get the integer coordinates for indexing
    coords = point_coords.long()
    # # We need to add a dimension for the channels
    # coords = torch.cat([torch.zeros_like(coords[..., :1]), coords], dim=-1)

    # Map the features back to the input feature map
    output[coords.split(1, dim=-1)] = features

    return output


class ChannelReducer(nn.Module):
    def __init__(self, num_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def binarize_feature_map_with_median_threshold(feature_map):
    feature_map = feature_map.to('cuda')
    median_val = 0.5
    binarized_feature_map = (feature_map > median_val).float()

    return binarized_feature_map


def get_thresholded_points(feature_map, threshold1, threshold2):
    B = feature_map.shape[0]
    feature_map = feature_map.mean(dim=1, keepdim=True)  # 将feature map的通道数降为1
    
    all_coords = []
    all_coords1 = []
    all_coords2 = []
    
    for b in range(B):
        feature_map_b = feature_map[b, 0]  # 去掉batch、channel维度
        
        coords = torch.stack(((feature_map_b > threshold1) & (feature_map_b < threshold2)).nonzero(as_tuple=True))
        coords1 = torch.stack((feature_map_b > threshold1).nonzero(as_tuple=True))
        coords2 = torch.stack((feature_map_b < threshold2).nonzero(as_tuple=True))

        coords = coords.permute(1, 0).unsqueeze(0)
        coords1 = coords1.permute(1, 0).unsqueeze(0)
        coords2 = coords2.permute(1, 0).unsqueeze(0)
        
        all_coords.append(coords)
        all_coords1.append(coords1)
        all_coords2.append(coords2)

    return all_coords1, all_coords2, all_coords

def extract_features_with_thresholds(input, feature_map, threshold1, threshold2):
    B = input.shape[0]
    # 获取三种像素坐标
    coords1_list, coords2_list, coords_list = get_thresholded_points(feature_map, threshold1, threshold2)

    output1_list = []
    output2_list = []
    output_list = []
    
    for b in range(B):
        # 根据像素坐标提取对应像素
        output1 = point_sample(input[b:b+1], coords1_list[b])
        output2 = point_sample(input[b:b+1], coords2_list[b])
        output = point_sample(input[b:b+1], coords_list[b])
        
        output1_list.append(output1)
        output2_list.append(output2)
        output_list.append(output)

    return output1_list, output2_list, output_list, coords1_list, coords2_list, coords_list

def point_sample(input, point_coords):

    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = torch.nn.functional.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=True, mode='bilinear', padding_mode='zeros')
    if add_dim:
        output = output.squeeze(3)
    return output

def Dual_Driving(AmbiguousPixel, DrivenPixel):
    # 调整形状以适应矩阵乘法
    Q = AmbiguousPixel.permute(1, 0)
    K = DrivenPixel
    V = DrivenPixel.permute(1, 0)
    vascular_scores = torch.matmul(Q, K)
    # softmax获取权重
    vascular_weights = F.softmax(vascular_scores, dim=-1)
    output = torch.matmul(vascular_weights, V)
    # 将输出的形状调整回原来的形状
    output = output.permute(1, 0)
    return output

class Pixel_Extractor(nn.Module):
    def __init__(self):
        super(Pixel_Extractor, self).__init__()

    def forward(self, feature_map, threshold1, threshold2):
        B = feature_map.shape[0]
        # Ensure the feature map is on the GPU
        feature_map = feature_map.to('cuda')
        feature_map_min = feature_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        feature_map_max = feature_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # Normalize feature_map
        feature_map_norm = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)

        feature_map1_list, feature_map2_list, feature_map3_list, coords1_list, coords2_list, coords_list = extract_features_with_thresholds(
            feature_map, feature_map_norm, threshold1, threshold2)

        # 将列表转换为tensor，处理空tensor的情况
        if all(fm.numel() > 0 for fm in feature_map1_list):
            feature_map1 = torch.cat(feature_map1_list, dim=0)
        else:
            feature_map1 = torch.empty(B, feature_map.shape[1], 0, device=feature_map.device)
            
        if all(fm.numel() > 0 for fm in feature_map2_list):
            feature_map2 = torch.cat(feature_map2_list, dim=0)
        else:
            feature_map2 = torch.empty(B, feature_map.shape[1], 0, device=feature_map.device)
            
        if all(fm.numel() > 0 for fm in feature_map3_list):
            feature_map3 = torch.cat(feature_map3_list, dim=0)
        else:
            feature_map3 = torch.empty(B, feature_map.shape[1], 0, device=feature_map.device)

        return feature_map1, feature_map2, feature_map3, coords1_list, coords2_list, coords_list


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1).to(torch.device("cuda:0"))
        self.gn = nn.GroupNorm(out_ch // 4, out_ch).to(torch.device("cuda:0"))
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True).to(torch.device("cuda:0"))
    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        return x




class Continuity_Perception(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Continuity_Perception, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x_out, threshold1, threshold2):
        B, C, H, W = x_out.shape
        # 分别归一化每一条通道
        x_min = x_out.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x_out.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_normalized = (x_out - x_min) / (x_max - x_min)

        denorm_value = x_min
        
        for b in range(B):  # 遍历每个batch
            for channel in range(C):
                # 对于每条通道的各个区域进行检测
                patches = F.unfold(x_normalized[b:b+1, channel:channel + 1], kernel_size=3, padding=1)
                patches = patches.permute(0, 2, 1).view(-1, H, W, 9)

                # 统计周围像素的类别
                center_pixel_condition = (patches[..., 4] > threshold1) & (patches[..., 4] < threshold2)
                surrounding_condition = patches[..., [0, 1, 2, 3, 5, 6, 7, 8]] < threshold1
                surrounding_condition = surrounding_condition.all(dim=-1)
                condition = (center_pixel_condition & surrounding_condition).unsqueeze(1)

                denorm_value_channel = denorm_value[b:b+1, channel:channel + 1, :, :]
                x_out[b:b+1, channel:channel+1][condition] = denorm_value_channel

        return x_out


class UncertaintyCheck(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UncertaintyCheck, self).__init__()
        self.conv_reduce = nn.Conv2d(input_channels, 1, kernel_size=1).to('cuda')
        self.conv_expand = nn.ConvTranspose2d(1, output_channels, kernel_size=1).to('cuda')

    def forward(self, uncertainty_vessel, uncertainty_background, x_out, uncertainty_coords, threshold2):
        B = x_out.shape[0]
        
        # Normalize
        uncertainty_vessel_normalized = (uncertainty_vessel - uncertainty_vessel.min()) / (uncertainty_vessel.max() - uncertainty_vessel.min())
        uncertainty_background_normalized = (uncertainty_background - uncertainty_background.min()) / (uncertainty_background.max() - uncertainty_background.min())

        # Reduce channel dimensions to 1
        uncertainty_vessel_reduced = torch.mean(uncertainty_vessel_normalized, dim=1, keepdim=True)
        uncertainty_background_reduced = torch.mean(uncertainty_background_normalized, dim=1, keepdim=True)

        # Reduce x_out channel dimension to 1
        x_out_reduced = self.conv_reduce(x_out)
        x_min = x_out_reduced.min()
        x_max = x_out_reduced.max()

        # 处理每个batch
        for b in range(B):
            if len(uncertainty_coords[b]) > 0 and uncertainty_coords[b].numel() > 0:
                coords_indexing = uncertainty_coords[b].squeeze(0).long()
                
                # 两种条件满足其一即认为是血管像素
                condition_mask = (uncertainty_vessel_reduced[b, 0] > threshold2) | (
                            uncertainty_background_reduced[b, 0] > threshold2)
                
                if coords_indexing.numel() > 0:
                    x_out_reduced[b, 0, coords_indexing[:, 0], coords_indexing[:, 1]][condition_mask] = x_max

        # Restore the channel dimension of x_out
        x_out_restored = self.conv_expand(x_out_reduced)

        return x_out_restored

# manba块输入输出维度不变
class MambaLayer_ambiguous_scan(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vessel_norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # 输入数据维度，也就是通道数
            d_state=d_state,  # SSM state expansion factor扩展因子
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor “todo”
        )
        self.vessel_mamba = Mamba(
            d_model=dim,  # 输入数据维度，也就是通道数
            d_state=d_state,  # SSM state expansion factor扩展因子
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor “todo”
        )
        self.threshold1 = nn.Parameter(torch.tensor(0.45))
        self.threshold2 = nn.Parameter(torch.tensor(0.55))
        self.threshold1.data.clamp_(0.4, 0.5)
        self.threshold2.data.clamp_(0.5, 0.6)
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]  # pytorch tensor shape: [B, C, H, W]

        assert C == self.dim
        n_tokens = x.shape[2:].numel()  # numel()函数返回张量元素个数
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(x_flat)
        x_norm = self.norm(x_mamba)
        x_out = x_norm.transpose(-1, -2).reshape(B, C, *img_dims)
        x_mamba = x_out

        # #检测特征图中被血管包围的不确定点
        check_scan = Continuity_Perception(C, C)
        x_out = check_scan(x_out, self.threshold1, self.threshold2)

        # #ADDR扫描部分
        extractor = Pixel_Extractor()
        (out_mamba_vessel, out_mamba_background, out_mamba_uncertainty, vessel_coords,
                    background_coords, uncertainty_coords) = extractor(x_out, self.threshold1, self.threshold2)
        
        # 检查是否有不确定像素
        has_uncertainty = any(coords.numel() > 0 for coords in uncertainty_coords if len(coords) > 0)
        if not has_uncertainty: 
            return x_out

        # Vessel/Background Driven - 处理每个batch
        uncertainty_vessel_list = []
        uncertainty_background_list = []
        
        for b in range(B):
            if len(out_mamba_uncertainty) > b and out_mamba_uncertainty[b].numel() > 0:
                out_mamba_vessel_b = out_mamba_vessel[b] if len(out_mamba_vessel) > b else torch.empty_like(out_mamba_uncertainty[b])
                out_mamba_background_b = out_mamba_background[b] if len(out_mamba_background) > b else torch.empty_like(out_mamba_uncertainty[b])
                
                uncertainty_vessel_b = Dual_Driving(out_mamba_uncertainty[b], out_mamba_vessel_b)
                uncertainty_background_b = Dual_Driving(out_mamba_uncertainty[b], out_mamba_background_b)
                
                uncertainty_vessel_list.append(uncertainty_vessel_b.unsqueeze(0))
                uncertainty_background_list.append(uncertainty_background_b.unsqueeze(0))
            else:
                # 创建空tensor
                empty_tensor = torch.empty(1, C, 0, device=x.device)
                uncertainty_vessel_list.append(empty_tensor)
                uncertainty_background_list.append(empty_tensor)
        
        if uncertainty_vessel_list:
            uncertainty_vessel = torch.cat(uncertainty_vessel_list, dim=0)
            uncertainty_background = torch.cat(uncertainty_background_list, dim=0)
        else:
            return x_out
            
        # #检查Driven后的血管像素
        check_uncertainty = UncertaintyCheck(C, C)
        x_out = check_uncertainty(uncertainty_vessel, uncertainty_background, x_out, uncertainty_coords, self.threshold2)
        x_out = (x_out - x_out.min()) / (x_out.max() - x_out.min())
        x_out = x_out * x_mamba
        
        # 最后mamba块处理
        x_out = x_out.reshape(B, C, n_tokens).transpose(-1, -2)
        x_out = self.vessel_mamba(x_out)
        x_out = self.vessel_norm(x_out)
        x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims) + x_mamba
        return x_out



class SerpScan(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str | torch.device = "cuda",
    ):


        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)
        padding_size = kernel_size // 2
        self.bn = nn.BatchNorm2d(kernel_size).to(device)
        # self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size).to(device)
        self.tanh = nn.Tanh().to(device)
        self.offset_conv = nn.Conv2d(in_channels, kernel_size, kernel_size=kernel_size, stride=(kernel_size, 1),
                                                                                            padding=(0, padding_size)).to(device)


    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset_y = self.offset_conv(input)
        offset_y = self.bn(offset_y)
        offset_y = self.tanh(offset_y)

        offset_x = self.offset_conv(input.permute(0, 1, 3, 2))
        offset_x = self.bn(offset_x)
        offset_x = self.tanh(offset_x)
        offset = torch.cat([offset_y, offset_x], dim=1)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature, grid = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        return deformed_feature, grid


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str | torch.device = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape

    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)



    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """

        y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
        y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

        x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
        x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        batch_size, _, height, width = offset.shape
        y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
        y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

        x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
        x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_offset_new_ = x_offset_new_.permute(0, 1, 3, 2)
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature, grid


import torch.nn.functional as F


def map_deformed_to_input(
        deformed_feature: torch.Tensor,
        input_feature: torch.Tensor,
        grid: torch.Tensor,
        interpolate_mode: str = "bilinear",
):
    """Map the deformed feature back to the input feature based on the coordinate map.

    Args:
        deformed_feature: feature that has been deformed with shape [B, C, K_H * H, K_W * W]
        input_feature: original input feature with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic'. Defaults to 'bilinear'.

    Return:
        combined_feature: input feature with deformed feature mapped back to it, with shape [B, C, H, W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")


    # Perform grid sampling with the deformed feature
    remapped_feature = F.grid_sample(
        input=deformed_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    # Create a mask of where the grid sampling was valid
    mask = F.grid_sample(
        input=torch.ones_like(deformed_feature),
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )
    mask = (mask > 0).float()  # Binary mask
    remapped_feature, input_feature = pad_to_match_dimensions(remapped_feature, input_feature)
    mask, input_feature = pad_to_match_dimensions(mask, input_feature)
    # Combine the original input feature and the remapped feature based on the mask
    combined_feature = mask * remapped_feature + (1 - mask) * input_feature

    return combined_feature


def _coordinate_map_scaling(coordinate_map, origin):
    """Scales the coordinate map values to a given range.

    Args:
        coordinate_map: The coordinate map to be scaled.
        origin: The target range [min, max].

    Returns:
        Scaled coordinate map.
    """
    min_val, max_val = origin
    coordinate_map = (coordinate_map - min_val) / (max_val - min_val) * 2 - 1
    return coordinate_map

def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


def pad_to_match_dimensions(tensor1, tensor2):
    """Pad the tensors to make their shapes the same either in height or width.

    Args:
        tensor1: A tensor with shape [B, C, H1, W1].
        tensor2: A tensor with shape [B, C, H2, W2].

    Returns:
        padded_tensor1: tensor1 padded to match the height or width of tensor2.
        padded_tensor2: tensor2 padded to match the height or width of tensor1.
    """
    B1, C1, H1, W1 = tensor1.shape
    B2, C2, H2, W2 = tensor2.shape

    # Ensure B and C dimensions are the same
    assert B1 == B2 and C1 == C2, "The B and C dimensions must be the same."

    # Determine padding for height

    max_height = max(H1, H2)
    max_width = max(W1, W2)
    if H1 < max_height:
        padding = (0, 0, 0, max_height - H1)
        tensor1 = F.pad(tensor1, padding, "constant", 0)
    if H2 < max_height:
        padding = (0, 0, 0, max_height - H2)
        tensor2 = F.pad(tensor2, padding, "constant", 0)

    # Determine padding for width
    if W1 < max_width:
        padding = (0, max_width - W1, 0, 0)  # Pad the last dimension (width)
        tensor1 = F.pad(tensor1, padding, "constant", 0)
    if W2 < max_width:
        padding = (0, max_width - W2, 0, 0)
        tensor2 = F.pad(tensor2, padding, "constant", 0)

    return tensor1, tensor2


class MambaLayer_Serpentine_Scan(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # 输入数据维度，也就是通道数
            d_state=d_state,  # SSM state expansion factor扩展因子
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor “todo”
        )
        self.gn = nn.GroupNorm(dim // 4, dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dsc_conv_x = nn.Conv2d(
            dim,
            dim,
            kernel_size=(9, 1),
            stride=(9, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=0,
        )
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]  # pytorch tensor shape: [B, C, H, W]
        n_tokens = x.shape[2:].numel()  # numel()函数返回张量元素个数
        img_dims = x.shape[2:]  # 图像维度HW
        assert C == self.dim
        device = x.device
        kernel_size = 9
        def process_x_direction(self, y, device):
            dsconv_x = SerpScan(self.dim, self.dim, kernel_size, 1, 0, True, device)
            dscx, grid = dsconv_x(y)
            B, C, H, W = dscx.shape
            chunks = H // kernel_size
            new_H = chunks * kernel_size
            dscx_reshaped = dscx[:, :, :new_H].view(B, C, chunks, kernel_size, W)
            dscx_reshaped = dscx_reshaped.view(B, C, chunks, -1)
            dscx_reshaped = dscx_reshaped.view(B, C, -1).transpose(-1, -2)
            dscx_reshaped = self.mamba(dscx_reshaped)
            dscx_reshaped = self.norm(dscx_reshaped)
            dscx_reshaped = dscx_reshaped.transpose(-1, -2).view(B, C, kernel_size, H * W // kernel_size)
            dscx_reshaped = ((dscx_reshaped.view(B, C, kernel_size, W, chunks)).permute(0, 1, 2, 4, 3)).reshape(B, C, -1, W)
            dscx_reshaped = map_deformed_to_input(dscx_reshaped, y, grid, "bilinear")
            return dscx_reshaped

        def process_y_direction(self, y, device):
            dsconv_y = SerpScan(self.dim, self.dim, kernel_size, 1, 1, True, device)
            dscy, grid = dsconv_y(y)
            B, C, H, W = dscy.shape
            n_tokens_y = dscy.shape[2:].numel()
            img_dims_y = dscy.shape[2:]
            dscy_flat = dscy.reshape(B, C, n_tokens_y).transpose(-1, -2)
            dscy_flat = self.mamba(dscy_flat)
            dscy_flat = self.norm(dscy_flat)
            dscy_flat = dscy_flat.transpose(-1, -2).reshape(B, C, *img_dims_y)
            dscy_flat = map_deformed_to_input(dscy_flat, y, grid, "bilinear")
            return dscy_flat

        def process_x_mamba(self, x, B, C, n_tokens, img_dims):
            x_out = x.reshape(B, C, n_tokens).transpose(-1, -2)  # Flatten and transpose
            x_out = self.mamba(x_out)  # Apply Mamba
            x_out = self.norm(x_out)  # Normalize
            x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)  # Transpose and reshape back
            return x_out

        def process_y_mamba(self, x, B, C, n_tokens, img_dims):
            y_out = x.permute(0, 1, 3, 2)  # Permute dimensions
            y_out = y_out.reshape(B, C, n_tokens).transpose(-1, -2)  # Flatten and transpose
            y_out = self.mamba(y_out)  # Apply Mamba
            y_out = self.norm(y_out)  # Normalize
            y_out = y_out.transpose(-1, -2).reshape(B, C, *img_dims)  # Transpose and reshape back
            y_out = y_out.permute(0, 1, 3, 2)  # Permute dimensions back
            return y_out

        x_out = process_x_mamba(self, x, B, C, n_tokens, img_dims)
        y_out = process_y_mamba(self, x, B, C, n_tokens, img_dims)
        dscx_reshaped = process_x_direction(self, x, device)
        dscy_reshaped = process_y_direction(self, x, device)
        fusion = EncoderConv(2 * C, C).to(device)
        out = fusion(torch.cat([dscx_reshaped, dscy_reshaped], 1)) + x_out + y_out
        return out


class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            stage = StackedResidualBlocks(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]
            if s == 5:
                mamba_layers.append(MambaLayer_ambiguous_scan(input_channels))
            else:
                mamba_layers.append(MambaLayer_Serpentine_Scan(input_channels))



        # self.stages = nn.Sequential(*stages)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.mamba_layers = nn.ModuleList(mamba_layers)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):

            x = self.stages[s](x)
            x = self.mamba_layers[s](x)


            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualMambaEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias)
            )
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                conv_bias=encoder.conv_bias,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                dropout_op=encoder.dropout_op,
                dropout_op_kwargs=encoder.dropout_op_kwargs,
                nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
            ))
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)): # s = 4 512

            x = self.transpconvs[s](lres_input)
            y = skips[-(s + 2)].to(x.device)

            x = torch.cat((x, y), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class SerpMamba(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                                            strides,
                                            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                            return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


def get_umamba_enc_from_plans(plans_manager: PlansManager,
                              dataset_json: dict,
                              configuration_manager: ConfigurationManager,
                              num_input_channels: int,
                              deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'SerpMamba'
    network_class = SerpMamba
    kwargs = {
        'SerpMamba': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_clas
ses=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == SerpMamba:
        model.apply(init_last_bn_before_add_to_0)

    return model








