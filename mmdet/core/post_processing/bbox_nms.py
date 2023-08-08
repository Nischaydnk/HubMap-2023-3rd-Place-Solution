# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ensemble_boxes import *
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    if nms_cfg["type"] == "weighted_cluster_nms":
        dets, labels, keep = weighted_cluster_nms(bboxes, scores, labels, nms_cfg)
    elif nms_cfg["type"] == "wbf":
        # dets, labels, keep = weighted_cluster_nms(bboxes, scores, labels, nms_cfg)
        dets, labels = wbf(bboxes, scores, labels, nms_cfg['iou_threshold'])
        return dets, labels
        
    else:
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (dets, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Dets are boxes with scores.
            Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs

def box_diou(boxes1, boxes2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    clt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    crb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    x1 = (boxes1[:, None, 0] + boxes1[:, None, 2]) / 2
    y1 = (boxes1[:, None, 1] + boxes1[:, None, 3]) / 2
    x2 = (boxes2[:, None, 0] + boxes2[:, None, 2]) / 2
    y2 = (boxes2[:, None, 1] + boxes2[:, None, 3]) / 2
    d = (x1 - x2.t()) ** 2 + (y1 - y2.t()) ** 2
    c = ((crb - clt) ** 2).sum(dim = 2)

    inter = (rb - lt).clamp(min = 0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter) - (d / c) ** 0.6 

def box_iou(boxes1, boxes2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min = 0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter) 


def wbf(bboxes, scores, labels, iou_thresh):
    # input:
    #   bboxes: tensor, Nx4, float32
    #   scores: tesnor, N, float32
    #   labels: tesnor, N, int64
    #   iou_threshold: float
    # output: 
    #   dets: tensor, Kx5, float32: x1, y1, x2, y2, score
    #   labels: tensor, K, int64
    device = bboxes.device
    boxes_list = bboxes.detach().cpu().numpy()
    scores_list = scores.detach().cpu().numpy()
    labels_list = labels.detach().cpu().numpy()
    X = np.max(boxes_list[:,[0,2]])
    Y = np.max(boxes_list[:,[1,3]])

    boxes_list[:,0] /= X 
    boxes_list[:,2] /= X 
    boxes_list[:,1] /= Y
    boxes_list[:,3] /= Y

    boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], 
            weights=None, iou_thr=iou_thresh)
    boxes[:,0] *= X 
    boxes[:,2] *= X 
    boxes[:,1] *= Y
    boxes[:,3] *= Y

    dets = np.concatenate([boxes, scores[:,None]], axis=1)
    dets = torch.tensor(dets, device=device, dtype=torch.float32)
    labels = torch.tensor(labels, device=device, dtype=torch.int64)
    return dets, labels
    
def weighted_cluster_nms(boxes, scores, idxs, nms_cfg):
    nms_cfg_ = nms_cfg.copy()
    n = len(scores)
    scores, idx = scores.sort(descending = True)
    boxes = boxes[idx]
    idxs = idxs[idx]

    class_agnostic = nms_cfg_.pop('class_agnostic', False)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    iou_threshold = nms_cfg_.pop("iou_threshold", 0.5)
    iou_method = nms_cfg_.pop("iou_method", "iou")
    iou_method = eval("box_" + iou_method)
    
    iou = iou_method(boxes_for_nms, boxes_for_nms).triu_(diagonal = 1)  # IoU矩阵，上三角化
    C = iou
    for _ in range(200):    
        A = C
        maxA = A.max(dim = 0)[0]   # 列最大值向量
        E = (maxA < iou_threshold).float().unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C) == True:     # 终止条件
            break
    keep = torch.arange(n)[maxA.cpu() < iou_threshold]  # 列最大值向量，二值化
    keep = keep.cuda()
    weights = (C * (C > iou_threshold).float() + torch.eye(n).cuda()) * (scores.reshape((1, n)))

    xx1 = boxes[:, 0].expand(n, n)
    yy1 = boxes[:, 1].expand(n, n)
    xx2 = boxes[:, 2].expand(n, n)
    yy2 = boxes[:, 3].expand(n, n)

    weightsum = weights.sum(dim = 1)         # 坐标加权平均
    xx1 = (xx1 * weights).sum(dim = 1) / weightsum
    yy1 = (yy1 * weights).sum(dim = 1) / weightsum
    xx2 = (xx2 * weights).sum(dim = 1) / weightsum
    yy2 = (yy2 * weights).sum(dim = 1) / weightsum
    boxes = torch.stack([xx1, yy1, xx2, yy2], 1)

    boxes = boxes[keep]
    scores = scores[keep]

    max_num = nms_cfg_.pop('max_num', -1)
    scores, idx = scores.sort(0, descending=True)
    
    if max_num > 0:
        keep = keep[:max_num]
        boxes = boxes[:max_num]
        scores = scores[:max_num]

    return torch.cat([boxes, scores[:, None]], -1), idxs, keep