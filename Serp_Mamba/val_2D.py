import numpy as np
import torch
from scipy.ndimage import zoom
from PIL import Image
import torch.nn.functional as F
from utils.metrics import calc_dice, calc_iou, calculate_metric_percase
# from distance_metrics_fast import hd95_fast


def calculate_metric_percase1(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    return calc_dice(pred, gt)

def calculate_metric_percase2(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    return calc_iou(pred, gt)

def test_single_volume_fast_SAM(image, label, net, classes, patch_size=[256, 256]):
    # 将输入图像和标签转换为numpy数组
    image, label = image.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy()

    # 初始化预测数组
    prediction = np.zeros_like(label)

    # 缩放图像以匹配网络输入尺寸
    def resize_image_and_label(image, output_size):
        x, y = image.shape[:2]
        if len(image.shape) == 2:
            image = zoom(image, (output_size[0] / x, output_size[1] / y), order=0)
        elif len(image.shape) == 3:
            image = zoom(image, (output_size[0] / x, output_size[1] / y, 1), order=0)
        return image

    zoomed_image = resize_image_and_label(image, patch_size)
    x, y = image.shape[:2]
    # 将处理后的图像转换为适合网络输入的格式
    input = torch.from_numpy(zoomed_image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        # Ensure the input image tensor has the correct dimensions
        if input.ndim == 5 and input.shape[-1] == 3:
            input = input.permute(0, 4, 1, 2, 3).squeeze(2)
        elif input.ndim == 4 and input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        image_embeddings = net.image_encoder(input)
        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_predictions = net.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=None,
        )

        # 插值回原始大小
        masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear",
                          align_corners=False)
        # 进行预测
        # output, _ = net(input, multimask_output=None)
        out = torch.argmax(torch.softmax(masks, dim=1), dim=1)
        # out = torch.sigmoid(net(input))
        out = out.cpu().detach().numpy().squeeze()
        # print("out,=",out.shape)
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
        # image = Image.fromarray(prediction.astype(np.uint8) * 255)
        # image.save('/home/ubuntu/project/Seg_code/output2_image.png')
        # image2 = Image.fromarray(label.astype(np.uint8) * 255)
        # image2.save('/home/ubuntu/project/Seg_code/output2_gt.png')

    # 计算并返回每个类别的评估指标
    metric_list1 = []
    metric_list2 = []
    for i in range(1, classes):
        metric_list1.append(calculate_metric_percase1(prediction == 1, label == 1))
        metric_list2.append(calculate_metric_percase2(prediction == 1, label == 1))

    return metric_list1, metric_list2
#2d
def test_image_fast(image, label, net, classes, patch_size=[256, 256]):
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
        # 进行预测
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1)
        # out = torch.sigmoid(net(input))
        out = out.cpu().detach().numpy().squeeze()
        # print("out,=",out.shape)
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
        # image = Image.fromarray(prediction.astype(np.uint8)*255)
        # image.save('/home/lishh237/Serp-Mamba/Seg_code/output2_image.png')
        # image2 = Image.fromarray(label.astype(np.uint8)*255)
        # image2.save('/home/lishh237/Serp-Mamba/Seg_code/output2_gt.png')

    # 计算并返回每个类别的评估指标
    metric_list1 = []
    metric_list2 = []
    for i in range(1, classes):

        metric_list1.append(calculate_metric_percase1(prediction == 1, label == 1))
        metric_list2.append(calculate_metric_percase2(prediction == 1, label == 1))
    
    return metric_list1, metric_list2



def test_single_volume_fast_multi_class(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    label = label.sum(dim=1)
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)

    ind_x = np.array([i for i in range(image.shape[0])])
    for ind in ind_x[::batch_size]:
        if ind + batch_size < image.shape[0]:

            stacked_slices = image[ind:ind + batch_size, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind:, ...] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_fast_multi_ouputs(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)

    ind_x = np.array([i for i in range(image.shape[0])])
    for ind in ind_x[::batch_size]:
        if ind + batch_size < image.shape[0]:

            stacked_slices = image[ind:ind + batch_size, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                output_list = net(input)
                for rater_id in range(len(output_list)):
                    out = torch.argmax(torch.softmax(
                        output_list[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                output_list = net(input)
                for rater_id in range(len(output_list)):
                    out = torch.argmax(torch.softmax(
                        output_list[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:, ...] = pred

    metric_array = np.zeros((label.shape[0], classes-1))

    for i in range(1, classes):
        for rater_id in range(label.shape[0]):
            metric_array[rater_id, i-1] = calculate_metric_percase(
                prediction[rater_id, ...] == i, label[rater_id, ...] == i)

    return metric_array


def test_single_volume_fast_padl(image, label, net, classes, patch_size=[256, 256], batch_size=24):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)

    ind_x = np.array([i for i in range(image.shape[0])])
    for ind in ind_x[::batch_size]:
        if ind + batch_size < image.shape[0]:

            stacked_slices = image[ind:ind + batch_size, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = net(
                    input, training=False)
                for rater_id in range(rater_samples.shape[0]):
                    out = torch.argmax(torch.softmax(
                        rater_samples[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]

            zoomed_slices = zoom(
                stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = net(
                    input, training=False)
                for rater_id in range(rater_samples.shape[0]):
                    out = torch.argmax(torch.softmax(
                        rater_samples[rater_id], dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[rater_id, ind:, ...] = pred

    metric_array = np.zeros((label.shape[0], classes-1))

    for i in range(1, classes):
        for rater_id in range(label.shape[0]):
            metric_array[rater_id, i-1] = calculate_metric_percase(
                prediction[rater_id, ...] == i, label[rater_id, ...] == i)

    return metric_array


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
