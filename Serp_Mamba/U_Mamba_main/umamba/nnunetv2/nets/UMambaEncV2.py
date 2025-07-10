import numpy as np

from torch import nn
from typing import Union, Type, List, Tuple

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
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from DSCnet.S3_DSConv_pro import DSConv_pro
from DSCnet.S3_DSCScan_pro import SerpScan, map_deformed_to_input
from DSCnet.S3_DSCNet_pro import EncoderConv
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


def save_image(image, save_path):
    out = torch.argmax(torch.softmax(image, dim=1), dim=1)
    # out = torch.sigmoid(net(input))
    # out = torch.sigmoid(image)
    out = out.cpu().detach().numpy().squeeze()
    # print("out,=",out.shape)

    # print("prediction =",prediction.shape)
    # print("prediction.min(),",prediction.min())
    # print("prediction.max(),",prediction.max())
    # print("label.min(),",label.min())
    # print("label.max(),",label.max())
    image = Image.fromarray(out.astype(np.uint8) * 255)
    image.save(save_path)

import cv2,os

def select_specific_channel(feature_map, channel_index):
    # 使用index_select函数选择指定通道
    feature_map_1channel = torch.index_select(feature_map, dim=1, index=torch.tensor([channel_index]).to('cuda'))

    return feature_map_1channel

def process_and_save_feature_map(feature_map, save_path, channel_index):
    # 选择指定通道
    feature_map_1channel = select_specific_channel(feature_map, channel_index).to('cuda')
    # 使用sigmoid函数进行归一化
    feature_map_1channel = torch.sigmoid(feature_map_1channel).to('cuda')

    # 将tensor转换为numpy数组，以便进行后续的图像处理
    feature_map_np = feature_map_1channel.squeeze().cpu().numpy()

    # 使用OpenCV进行直方图均衡化
    feature_map_np = (feature_map_np * 255).astype(np.uint8)  # 转换为8位无符号整型
    # 设置阈值
    thresh = 130
    # 二值化图像
    _, feature_map_np = cv2.threshold(feature_map_np, thresh, 255, cv2.THRESH_BINARY)
    # 修改保存路径以包含通道索引
    base_name, ext = os.path.splitext(save_path)
    save_path = f"{base_name}_channel_{channel_index}{ext}"
    # 保存图像
    cv2.imwrite(save_path, feature_map_np)

def process_and_save_all_channels(feature_map, save_path):
    num_channels = feature_map.shape[1]
    for channel_index in range(num_channels):
        process_and_save_feature_map(feature_map, save_path, channel_index)

class ChannelReducer(nn.Module):
    def __init__(self, num_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def binarize_feature_map_with_median_threshold(feature_map):
    # Ensure the feature map is on the GPU
    feature_map = feature_map.to('cuda')

    # Compute the median value
    median_val = 0.5

    # Binarize the feature map
    binarized_feature_map = (feature_map > median_val).float()

    return binarized_feature_map


def get_thresholded_points(feature_map, threshold1, threshold2):
    feature_map = feature_map.mean(dim=1, keepdim=True)  # 将feature map的通道数降为1
    feature_map = feature_map.squeeze(0).squeeze(0)  # 去掉batch、channel维度

    coords = torch.stack(((feature_map > threshold1) & (feature_map < threshold2)).nonzero(as_tuple=True))
    coords1 = torch.stack((feature_map > threshold1).nonzero(as_tuple=True))
    coords2 = torch.stack((feature_map < threshold2).nonzero(as_tuple=True))

    coords = coords.permute(1, 0).unsqueeze(0)
    coords1 = coords1.permute(1, 0).unsqueeze(0)
    coords2 = coords2.permute(1, 0).unsqueeze(0)

    return coords1, coords2, coords

def extract_features_with_thresholds(input, feature_map, threshold1, threshold2):

    coords1, coords2, coords = get_thresholded_points(feature_map, threshold1, threshold2)

    output1 = point_sample(input, coords1)
    output2 = point_sample(input, coords2)
    output = point_sample(input, coords)

    return output1, output2, output, coords1, coords2, coords

def point_sample(input, point_coords):

    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = torch.nn.functional.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=True, mode='bilinear', padding_mode='zeros')
    if add_dim:
        output = output.squeeze(3)
    return output

def self_attention(Q_feature, KV_feature):
    # 调整Q的形状为(H1W1 x C)
    Q = Q_feature.permute(1, 0)  # Q形状变为(H1W1, C)
    # K保持不变，形状为(C, H2W2)
    K = KV_feature  # K形状为(C, H2W2)
    # 调整V的形状为(H2W2 x C)
    V = KV_feature.permute(1, 0)  # V形状变为(H2W2, C)
    # 计算Q和K的点积，得到注意力得分矩阵
    attention_scores = torch.matmul(Q, K)  # 形状为(H1W1, H2W2)
    # 应用softmax获取注意力权重
    attention_weights = F.softmax(attention_scores, dim=-1)  # 形状为(H1W1, H2W2)
    # 使用注意力权重和V进行加权求和
    output = torch.matmul(attention_weights, V)  # 形状为(H1W1, C)
    # 将输出的形状调整回(C, H1W1)
    output = output.permute(1, 0)  # 输出形状为(C, H1W1)
    return output

class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()


    def forward(self, feature_map, threshold1, threshold2):
        # Ensure the feature map is on the GPU
        feature_map = feature_map.to('cuda')
        feature_map_min = feature_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        feature_map_max = feature_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # Normalize feature_map
        feature_map_norm = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)

        feature_map1, feature_map2, feature_map3, coords1, coords2, coords = extract_features_with_thresholds(
            feature_map, feature_map_norm, threshold1, threshold2)

        return feature_map1, feature_map2, feature_map3, coords1, coords2, coords

def DSConv_block(int_, y, device):
    dscon_in = EncoderConv(int_, int_).to(device)
    dscon_out = EncoderConv(3 * int_, int_).to(device)
    dsconv_x = DSConv_pro(int_, int_, 9, 1, 0, True, device)
    dsconv_y = DSConv_pro(int_, int_, 9, 1, 1, True, device)

    x_int = dscon_in(y)
    dscx = dsconv_x(y)
    dscy = dsconv_y(y)

    dsc_x = dscon_out(torch.cat([x_int, dscx, dscy], 1))

    return dsc_x



class ProcessXOut(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ProcessXOut, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x_out, threshold1, threshold2):
        # Normalize each channel of x_out independently
        x_min = x_out.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x_out.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_normalized = (x_out - x_min) / (x_max - x_min)

        # Assuming x_out shape is [batch_size, channels, height, width]
        _, C, H, W = x_out.shape
        denorm_value = x_min
        for channel in range(C):
            # Extract 3x3 patches for the current channel
            patches = F.unfold(x_normalized[:, channel:channel + 1], kernel_size=3, padding=1)
            patches = patches.permute(0, 2, 1).view(-1, H, W, 9)  # Reshape to match original shape

            # Compute conditions for the current channel
            center_pixel_condition = (patches[..., 4] > threshold1) & (patches[..., 4] < threshold2)
            surrounding_condition = patches[..., [0, 1, 2, 3, 5, 6, 7, 8]] < threshold1
            surrounding_condition = surrounding_condition.all(dim=-1)
            condition = (center_pixel_condition & surrounding_condition).unsqueeze(1)

            # Update x_out based on conditions for the current channel
            # Calculate denormalized value for pixels satisfying the condition

            denorm_value_channel = denorm_value[:, channel:channel + 1, :, :]
            x_out[:, channel:channel+1][condition] = denorm_value_channel

        # x_out is now normalized and updated, with its original channel count unchanged
        return x_out


class ProcessUncertaintyAndXOut(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ProcessUncertaintyAndXOut, self).__init__()
        self.conv_reduce = nn.Conv2d(input_channels, 1, kernel_size=1).to('cuda')
        self.conv_expand = nn.ConvTranspose2d(1, output_channels, kernel_size=1).to('cuda')

    def forward(self, uncertainty_vessel, uncertainty_background, x_out, uncertainty_coords, threshold2):
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

        # Modify x_out based on uncertainty_coords
        coords_indexing = uncertainty_coords.squeeze(0).long()

        # Create condition mask
        condition_mask = (uncertainty_vessel_reduced[0, 0] > threshold2) | (
                    uncertainty_background_reduced[0, 0] > threshold2)
        # Use advanced indexing to update x_out_reduced where the condition is True
        # Expand dims of denorm_val to match the broadcasting requirements
        x_out_reduced[0, 0, coords_indexing[:, 0], coords_indexing[:, 1]][condition_mask] = x_max

        # Restore the channel dimension of x_out
        x_out_restored = self.conv_expand(x_out_reduced)

        return x_out_restored

# manba块输入输出维度不变
class MambaLayer_DSC_withscan(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vessel_norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.vessel_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.threshold1 = nn.Parameter(torch.tensor(0.45))
        self.threshold2 = nn.Parameter(torch.tensor(0.55))
        self.threshold1.data.clamp_(0.4, 0.5)
        self.threshold2.data.clamp_(0.5, 0.6)
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # 将图像展平，转置 (B, n_tokens, C)
        x_mamba = self.mamba(x_flat)
        x_norm = self.norm(x_mamba)
        x_out = x_norm.transpose(-1, -2).reshape(B, C, *img_dims)  # 将输出转置，reshape回原图像形状
        x_mamba = x_out

        # #提前检测特征图中被血管包围的不确定点并进行操作
        check_scan = ProcessXOut(C, C)
        x_out = check_scan(x_out, self.threshold1, self.threshold2)

        # #ADDR扫描部分
        binarizer = Binarizer()
        (out_mamba_vessel, out_mamba_background, out_mamba_uncertainty, vessel_coords,
                    background_coords, uncertainty_coords) = binarizer(x_out, self.threshold1, self.threshold2)
        n_tokens_ = out_mamba_uncertainty.shape[2:].numel()
        # 如果全部像素是背景，直接返回
        if n_tokens_ == 0: return x_out

        # #Vessel/Background Driven
        out_mamba_vessel = out_mamba_vessel.squeeze(0)
        out_mamba_background = out_mamba_background.squeeze(0)
        out_mamba_uncertainty = out_mamba_uncertainty.squeeze(0)
        uncertainty_vessel = (self_attention(out_mamba_uncertainty, out_mamba_vessel)).unsqueeze(0)
        uncertainty_background = (self_attention(out_mamba_uncertainty, out_mamba_background)).unsqueeze(0)
        # #检查Driven后的血管像素
        check_uncertainty = ProcessUncertaintyAndXOut(C, C)
        x_out = check_uncertainty(uncertainty_vessel, uncertainty_background, x_out, uncertainty_coords, self.threshold2)
        x_out = (x_out - x_out.min()) / (x_out.max() - x_out.min())
        x_out = x_out * x_mamba
        # vessel scan
        x_out = x_out.reshape(B, C, n_tokens).transpose(-1, -2)
        x_out = self.vessel_mamba(x_out)
        x_out = self.vessel_norm(x_out)
        x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims) + x_mamba
        return x_out

class Space_layer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        return x

class MambaLayer_DSC_noscan(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # self.mamba_x = Mamba(
        #     d_model=dim,  # 输入数据维度，也就是通道数
        #     d_state=d_state,  # SSM state expansion factor扩展因子
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor “todo”
        # )
        # self.mamba_y = Mamba(
        #     d_model=dim,  # 输入数据维度，也就是通道数
        #     d_state=d_state,  # SSM state expansion factor扩展因子
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor “todo”
        # )
        # self.mamba1 = Mamba(
        #     d_model=dim,  # 输入数据维度，也就是通道数
        #     d_state=d_state,  # SSM state expansion factor扩展因子
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor “todo”
        # )
        # self.mamba2 = Mamba(
        #     d_model=dim,  # 输入数据维度，也就是通道数
        #     d_state=d_state,  # SSM state expansion factor扩展因子
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor “todo”
        # )
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
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
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
            x_out = self.mamba(x_out)
            x_out = self.norm(x_out)
            x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)  # Transpose and reshape back
            return x_out

        def process_y_mamba(self, x, B, C, n_tokens, img_dims):
            y_out = x.permute(0, 1, 3, 2)  # Permute dimensions
            y_out = y_out.reshape(B, C, n_tokens).transpose(-1, -2)  # Flatten and transpose
            y_out = self.mamba(y_out)
            y_out = self.norm(y_out)
            y_out = y_out.transpose(-1, -2).reshape(B, C, *img_dims)  # Transpose and reshape back
            y_out = y_out.permute(0, 1, 3, 2)  # Permute dimensions back
            return y_out

        x_out = process_x_mamba(self, x, B, C, n_tokens, img_dims)
        y_out = process_y_mamba(self, x, B, C, n_tokens, img_dims)
        dscx_reshaped = process_x_direction(self, x, device)
        dscy_reshaped = process_y_direction(self, x, device)
        fusion = EncoderConv(2 * C, C).to(device)
        out = fusion(torch.cat([dscx_reshaped, dscy_reshaped], 1)) + x_out + y_out
        # out = fusion(torch.cat([dscx_reshaped, dscy_reshaped], 1))
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
                mamba_layers.append(MambaLayer_DSC_withscan(input_channels))
            else:
                mamba_layers.append(MambaLayer_DSC_noscan(input_channels))


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
        ret = []  # 保存每个stage+mamba的输出

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

        stages = []
        transpconvs = []
        seg_layers = []
        addr_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias)
            )
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
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.addr_layers = nn.ModuleList(addr_layers)

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

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):

        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]


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


class UMambaEnc(nn.Module):
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

    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc
    kwargs = {
        'UMambaEnc': {
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
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == UMambaEnc:
        model.apply(init_last_bn_before_add_to_0)

    return model

