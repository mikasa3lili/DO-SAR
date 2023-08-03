import argparse
import glob
from pathlib import Path

#import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
#from visual_utils import visualize_utils as V
from pcdet.utils import box_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        print('root_path:',self.root_path, 'ext:',self.ext)
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        print('data_file_list:',data_file_list)

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

@staticmethod
def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
    def get_template_prediction(num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def generate_single_sample_dict(batch_index, box_dict):
        pred_scores = box_dict['pred_scores'].cpu().numpy()
        pred_boxes = box_dict['pred_boxes'].cpu().numpy()
        pred_labels = box_dict['pred_labels'].cpu().numpy()
        pred_dict = get_template_prediction(pred_scores.shape[0])
        if pred_scores.shape[0] == 0:
            return pred_dict

        calib = batch_dict['calib'][batch_index]
        image_shape = batch_dict['image_shape'][batch_index]
        pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, calib, image_shape=image_shape
        )

        pred_dict['name'] = np.array(class_names)[pred_labels - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        pred_dict['bbox'] = pred_boxes_img
        pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        pred_dict['location'] = pred_boxes_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        pred_dict['score'] = pred_scores
        pred_dict['boxes_lidar'] = pred_boxes

        return pred_dict

    annos = []
    for index, box_dict in enumerate(pred_dicts):
        frame_id = batch_dict['frame_id'][index]

        single_pred_dict = generate_single_sample_dict(index, box_dict)
        single_pred_dict['frame_id'] = frame_id
        annos.append(single_pred_dict)

        if output_path is not None:
            cur_det_file = output_path / ('%s.txt' % frame_id)
            with open(cur_det_file, 'w') as f:
                bbox = single_pred_dict['bbox']
                loc = single_pred_dict['location']
                dims = single_pred_dict['dimensions']  # lhw -> hwl

                for idx in range(len(bbox)):
                    print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                             bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                             dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                             loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                             single_pred_dict['score'][idx]), file=f)

    return annos

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print('data_dict:', data_dict.keys(),'pred_dict:',pred_dicts.keys())
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.show(stop=True)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()
