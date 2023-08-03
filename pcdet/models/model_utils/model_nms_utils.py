import torch
from ...ops.iou3d_nms import iou3d_nms_utils
import cv2
import os

#def class_agnostic_nms(id, box2d, box_preds_2d, box_scores, box_preds, nms_config, score_thresh=None):
def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    #src_box_preds_2d = box_preds_2d
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
        #box_preds_2d = box_preds_2d[scores_mask.cpu()]
    selected = []

    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config)
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]] #train:512  test:100

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]

    # ------------my------------
    # box_preds_2d = torch.from_numpy(box_preds_2d[selected.cpu()])
    # if len(box_preds_2d.shape) == 1 :
    #     box_preds_2d = box_preds_2d.unsqueeze(0)
    # box_scores_inds = src_box_scores[selected.cpu()]
    # if box_preds_2d.shape[0] != 0:
    #     iou2d = image_box_overlap(box_preds_2d, box2d)
    #     max_overlaps, gt_assignment = torch.max(iou2d, dim=1)
    #     pos_inds = (max_overlaps >= 0.7).float()
    #     #print('pos_inds:',pos_inds,pos_inds.shape,'pos2:',pos2,'pos2_float:',pos2.float())#, 'box_scores_inds:',box_scores_inds.shape, 'box_scores_inds[fg_inds]:',box_scores_inds[pos_inds].shape)
    #     new_2d = box2d[gt_assignment.long()]
    #     new_2d[:,4] *= pos_inds
    #
    #     score_23d = new_2d[:,4].cpu() + box_scores_inds.cpu()
    #     score_pos = score_23d>0.3
    #     selected = selected[score_pos]
    #
    #     new_3d = torch.from_numpy(src_box_preds_2d[selected.cpu()])
    #     box_scores_inds = src_box_scores[selected.cpu()]
    #     #if new_3d.shape[0] != 0:
    #     # for i in range(new_2d.shape[0]):
    #     #     tt1 = (int(new_2d[i][0]), int(new_2d[i][1]))
    #     #     tt2 = (int(new_2d[i][2]), int(new_2d[i][3]))
    #     #     cv2.rectangle(img, tt1, tt2, (0, 0, 255), 1)
    #     #     a = round(new_2d[i][4].item(), 2)
    #     #     cv2.putText(img, str(a), tt1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)  # red
    #
    #     # if len(new_3d.shape) == 1:
    #     #     new_3d = new_3d.unsqueeze(0)
    #     # for j in range(new_3d.shape[0]):
    #     #     t1 = (int(new_3d[j][0]), int(new_3d[j][1]))
    #     #     t2 = (int(new_3d[j][2]), int(new_3d[j][3]))
    #     #     cv2.rectangle(img, t1, t2, (0, 255, 0), 1)
    #     #     b = round(box_scores_inds[j].item(), 2)
    #     #     ab = round(score_23d[j].item(), 2)
    #     #     cv2.putText(img, str(b), t1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)  # green
    #     #     cv2.putText(img, str(ab), t2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    #     # path = os.path.join("/data/hx_1/SASA-2/img-2d3d/" + str(id) + '.png')
    #     # cv2.imwrite(path, img)
    #
    # inds = src_box_scores[selected]>0.15
    # selected = selected[inds]
    # #print('selected:',selected, selected.shape, 'src:',src_box_scores[selected], src_box_scores[selected].shape)

    return selected, src_box_scores[selected]

def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]  # ! gt boxes的总数
    K = query_boxes.shape[0]  # ! 要检测的图像总数
    overlaps = boxes.new(N, K).zero_()
    for k in range(K):  # 遍历要检测的图像总数
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))  # ! 第k个dt框的面积
        for n in range(N):  # 遍历gt boxes
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:  # 如果宽度方向有重叠
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:  # 如果高度方向有重叠
                    if criterion == -1:  # 默认执行criterion = -1
                        ua = (  # 总的面积
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:  # 当检测图像包含gt box时
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))  # gt box的面积
                    elif criterion == 1:  # 当gt box包含检测图像时
                        ua = qbox_area  # 检测图像的面积
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua  # ! 计算iou
    return overlaps

def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
