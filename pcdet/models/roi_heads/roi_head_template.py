import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from ...datasets.kitti.kitti_dataset import KittiDataset
from ...utils import box_utils, calibration_kitti

class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

        self.root_split_path = '../data/kitti/training'

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict, rcnn_cls_conf):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        tb_dict = {}

        id = forward_ret_dict['id']
        gt_2d = forward_ret_dict['gt_2d']  #(B, 128, 4)
        gt_2d_aug = forward_ret_dict['gt_2d_aug']
        img = forward_ret_dict['img']
        a = torch.ones(1, 4).cuda()
        c = torch.zeros(1, 4).cuda()
        for i in range(id.shape[0]):
            calib = self.get_calib(id[i])
            rcnn_reg_i = roi_boxes3d[i].squeeze(0)
            rcnn_reg_2d = box_utils.boxes3d_lidar_to_kitti_camera(rcnn_reg_i.cpu().detach(), calib)
            rcnn_reg_2d = box_utils.boxes3d_kitti_camera_to_imageboxes(rcnn_reg_2d, calib)  #(Bx128, 4)
        #     tar_gt = gt_2d[i]
        #     roi_2d = torch.from_numpy(rcnn_reg_2d).cuda()
        #     iou2d = self.image_box_overlap(tar_gt.float(), roi_2d.float())
        #     max_overlaps, gt_assignment = torch.max(iou2d, dim=1)
        #     fg_inds = torch.nonzero((max_overlaps >= 0.3) & (max_overlaps <= 1)).view(-1)
        #
        #     now_roi = roi_2d[gt_assignment[fg_inds].long()]
        #     now_gt = tar_gt[fg_inds]
        #     img_draw = img[i].cpu().numpy()
        #     now_roi = rcnn_reg_2d
        #     for b in range(rcnn_reg_2d.shape[0]):
        #         tt1 = (int(now_roi[b][0]), int(now_roi[b][1]))
        #         tt2 = (int(now_roi[b][2]), int(now_roi[b][3]))
        #         # pp1 = (int(now_gt[b][0]), int(now_gt[b][1]))
        #         # pp2 = (int(now_gt[b][2]), int(now_gt[b][3]))
        #         cv2.rectangle(img_draw, tt1, tt2, (255, 0, 255), 1)
        #         #cv2.rectangle(img_draw, pp1, pp2, (255, 0, 0), 1)
        #     path = os.path.join("/data/hx_1/SASA/img/" + str(id[i]) + '.png')
        #     cv2.imwrite(path, img_draw)
        #
        #     if tar_gt[fg_inds].shape[0] != 0:
        #         a = torch.cat((a, now_roi), dim=0)
        #         c = torch.cat((c, now_gt), dim=0)
        # loss_b = self.iou_loss(a, c)
        # print('loss_b:',loss_b)
        # inds = loss_b<0.999
        # loss_b = loss_b[inds]
        # if loss_b.shape[0] > 0:
        #     rcnn_loss_reg_23d = loss_b.sum()/loss_b.shape[0]
        # else:
        #     rcnn_loss_reg_23d = (0.1*torch.ones(1)).sum().cuda()
        # print('loss_23d:',rcnn_loss_reg_23d)

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),# [1, Bx128, 7]
                reg_targets.unsqueeze(dim=0),)  # [B, M, 7]         # [1, Bx128, 7]
#--------------------my---------------------------
            rcnn_cls_conf = rcnn_cls_conf.unsqueeze(-1).repeat(1, 7).unsqueeze(0)
            rcnn_loss_reg = 4*rcnn_cls_conf * rcnn_loss_reg
#-------------------------------------------------
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        #rcnn_loss_reg += rcnn_loss_reg_23d
        #tb_dict['rcnn_loss_reg_23d'] = rcnn_loss_reg_23d.item()
        return rcnn_loss_reg, tb_dict

    def get_calib(self, idx):
        calib_file = os.path.join(self.root_split_path , 'calib/' + ('%s.txt' % idx))
        #assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_cls_conf = torch.sigmoid(rcnn_cls.view(-1))
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, rcnn_cls_conf, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, rcnn_cls_conf, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict, rcnn_cls_conf)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:
        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds

    def iou_loss(self, preds, targets):
        lt_min = torch.min(preds[:, :2], targets[:, :2])
        rb_min = torch.min(preds[:, 2:], targets[:, 2:])
        wh_min = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]
        area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        union = (area1 + area2 - overlap)
        iou = overlap / union
        iou = torch.clamp(iou, 0.0001, 0.9999)
        return -torch.log(iou)

    def image_box_overlap(self, boxes, query_boxes, criterion=-1):
        N = boxes.shape[0]  # ! gt boxes的总数
        K = query_boxes.shape[0]  # ! 要检测的图像总数
        # overlaps = np.zeros((N, K), dtype=boxes.dtype) #! np类型 float
        overlaps = boxes.new(N, K).zero_()
        for k in range(K):  # 遍历要检测的图像总数
            qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                         (query_boxes[k, 3] - query_boxes[k, 1]))  # ! 第k个dt框的面积
            for n in range(N):  # 遍历gt boxes
                iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                    boxes[n, 0], query_boxes[k, 0]))
                # 重叠部分的宽度 = 两个图像右边缘的较小值 - 两个图像左边缘的较大值
                if iw > 0:  # 如果宽度方向有重叠
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                        boxes[n, 1], query_boxes[k, 1]))
                    # 重叠部分的高度 = 两个图像上边缘的较小值 - 两个图像下边缘的较大值
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
