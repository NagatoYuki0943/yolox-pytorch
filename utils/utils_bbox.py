import numpy as np
import torch
from torchvision.ops import nms, boxes

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

#------------------------#
#   解码预测数据
#------------------------#
def decode_outputs(outputs, input_shape):
    """
    outputs:        输入前代表每个特征层的预测结果
        batch_size, 5 + num_classes, 80, 80
        batch_size, 5 + num_classes, 40, 40
        batch_size, 5 + num_classes, 20, 20
    input_shape:    输入图像大小 640 640

    """
    grids   = []    # 每个特征层对应的网格点
    strides = []    # 每个特征层对应的步长
    # 获得后面的宽高    [[80, 80], [40, 40] [20, 20]]
    hw      = [x.shape[-2:] for x in outputs]
    #---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #
    #   flatten(start_dim=2) 将后面两个宽高铺平放到一起
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 5 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #
    #   三层堆叠到一起
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠,转置后为"""
    #       [batch_size, 8400, 5 + num_classes]
    #---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    #---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率和是否有物体的概率(只对分类概率进行处理,不对中心宽高参数处理)
    #   0~1之间,表示可能性
    #---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    #---------------------------------------------------#
    #   处理三层的宽高数据
    #---------------------------------------------------#
    for h, w in hw:
        #---------------------------#
        #   根据输入特征层的高宽生成网格点
        #---------------------------#
        grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
        #---------------------------#
        #   torch.stack() 拼接后会添加新维度, 2 就是在最后维度添加标记
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400,  2
        #---------------------------#
        grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape           = grid.shape[:2]    # [1,6400], [1,1600], [1,400]

        grids.append(grid)
        # 每个特征点对应的步长  640/80=8 /40=16 /20=32
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    """将网格点堆叠到一起"""
    #---------------------------#
    #   将网格点堆叠到一起
    #   [[1, 6400, 2], [1, 1600, 2], [1, 400,  2]] cat [1, 8400, 2]
    #   将每个网格点对应的步长堆叠到一起
    #   [[1, 6400, 1], [1, 1600, 1], [1, 400,  1]] cat [1, 8400, 1]
    #---------------------------#
    grids               = torch.cat(grids, dim=1).type(outputs.type())
    strides             = torch.cat(strides, dim=1).type(outputs.type())
    #---------------------------#
    #   根据网格点进行解码
    #   直接将预测值进行处理,不需要进行和v5一样的sigmoid处理
    #---------------------------#
    outputs[...,  :2]   = (outputs[..., :2] + grids) * strides      # outputs0,1 加上坐标,乘以步长 得到预测框中心点坐标
    outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides    # outputs2,3 的e指数,乘以步长  得到预测框宽高
    #---------------------------#
    #   归一化,除以宽高
    #---------------------------#
    outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]      # x w   数据是hw, 所以先用的[1]
    outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]      # y h
    #---------------------------#
    #   outputs: [batch_size, 8400, 5 + num_classes]
    #---------------------------#
    return outputs


"""非极大值抑制"""
def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角坐标的格式。
    #   prediction:   [batch_size, num_anchors, 5 + num_classes]
    #----------------------------------------------------------#
    box_corner           = prediction.new(prediction.shape)
    box_corner[:, :, 0]  = prediction[:, :, 0] - prediction[:, :, 2] / 2    # x - 1/2 w = x1
    box_corner[:, :, 1]  = prediction[:, :, 1] - prediction[:, :, 3] / 2    # y - 1/2 h = y1
    box_corner[:, :, 2]  = prediction[:, :, 0] + prediction[:, :, 2] / 2    # x + 1/2 w = x2
    box_corner[:, :, 3]  = prediction[:, :, 1] + prediction[:, :, 3] / 2    # y + 1/2 h = y2
    prediction[:, :, :4] = box_corner[:, :, :4]                             # 替换前4个数据换成左上角右下角的格式

    output = [None for _ in range(len(prediction))]
    #----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   image_pred[:, 5:5 + num_classes] 取出分类信息
        #   class_conf  [8400, 1]     种类置信度(数字)
        #   class_pred  [8400, 1]     种类(下标)
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选,返回0/1
        #   image_pred[:, 4] * class_conf[:, 0]  是否包含物体 * 置信度 得到最后的置信度,通过比较为 False 或 True
        #   [8400, 1] * [8400, 1] = [8400]
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   堆叠位置参数,是否有物体,种类置信度,种类
        #   detections  [8400, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf(是否包含物体置信度), class_conf(种类置信度), class_pred(种类预测值)
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        #------------------------------------------#
        #   初步筛选,减少框的数量
        #   [8400, 7] -> [80, 7]
        #------------------------------------------#
        detections = detections[conf_mask]

        nms_out_index = boxes.batched_nms(
            detections[:, :4],                      # 坐标
            detections[:, 4] * detections[:, 5],    # obj_conf(是否包含物体置信度) * class_conf(种类置信度) 结果是1维数据
            detections[:, 6],                       # class_pred(种类预测值)
            nms_thres,
        )

        #------------------------------------------#
        #   nms进一步减少框的数量
        #   [80, 7] -> [10, 7]
        #------------------------------------------#
        output[i]   = detections[nms_out_index]

        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]

        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data

        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        # 去除图片灰条
        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    # [10, 7]
    return output
