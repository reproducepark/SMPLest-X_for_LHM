import os
import os.path as osp
import sys
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
import json
from tqdm import tqdm
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--save_json', action='store_true', help='Save SMPL-X results as JSON')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    json_folder = osp.join(root_dir, 'demo', 'output_json', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    if args.save_json:
        os.makedirs(json_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path = osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
        elif not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            # yolo_bbox = yolo_bbox[0]
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # For single person detection, use the first detected person
        if num_bbox > 0:
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[0][0]
            yolo_bbox_xywh[1] = yolo_bbox[0][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[0][2] - yolo_bbox[0][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[0][3] - yolo_bbox[0][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            # Extract SMPL-X parameters
            # Get SMPL-X parameters from model output
            smplx_root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
            smplx_body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0]
            smplx_lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0]
            smplx_rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0]
            smplx_jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0]
            smplx_shape = out['smplx_shape'].detach().cpu().numpy()[0]
            smplx_expr = out['smplx_expr'].detach().cpu().numpy()[0]
            cam_trans = out['cam_trans'].detach().cpu().numpy()[0]
            
            # Convert body pose from 63 to 21 parameters (3 per joint) as nested arrays
            body_pose_21 = smplx_body_pose[:63].reshape(-1, 3)[:21].tolist()
            
            # Convert hand poses from 45 to 15 parameters (3 per joint) as nested arrays
            lhand_pose_15 = smplx_lhand_pose[:45].reshape(-1, 3)[:15].tolist()
            rhand_pose_15 = smplx_rhand_pose[:45].reshape(-1, 3)[:15].tolist()
            
            # Extract eye poses from expression parameters (assuming they're included)
            # If not available, use zeros as placeholder
            leye_pose = np.zeros(3)
            reye_pose = np.zeros(3)
            
            frame_results = {
                'betas': smplx_shape.tolist(),
                'root_pose': smplx_root_pose.tolist(),
                'body_pose': body_pose_21,
                'jaw_pose': smplx_jaw_pose.tolist(),
                'leye_pose': leye_pose.tolist(),
                'reye_pose': reye_pose.tolist(),
                'lhand_pose': lhand_pose_15,
                'rhand_pose': rhand_pose_15,
                'trans': cam_trans.tolist(),
                'focal': cfg.model.focal,
                'princpt': cfg.model.princpt,
                'img_size_wh': [original_img_width, original_img_height],
                'pad_ratio': 0.2
            }
        else:
            # No person detected, create empty result
            frame_results = {
                'betas': [0.0] * 10,
                'root_pose': [0.0] * 3,
                'body_pose': [[0.0, 0.0, 0.0] for _ in range(21)],  # 21 joints * 3 = 63 parameters
                'jaw_pose': [0.0] * 3,
                'leye_pose': [0.0] * 3,
                'reye_pose': [0.0] * 3,
                'lhand_pose': [[0.0, 0.0, 0.0] for _ in range(15)],  # 15 joints * 3
                'rhand_pose': [[0.0, 0.0, 0.0] for _ in range(15)],  # 15 joints * 3
                'trans': [0.0] * 3,
                'focal': cfg.model.focal,
                'princpt': cfg.model.princpt,
                'img_size_wh': [original_img_width, original_img_height],
                'pad_ratio': 0.2
            }

        # Save JSON results
        if args.save_json:
            json_path = osp.join(json_folder, f'{int(frame):06d}.json')
            with open(json_path, 'w') as f:
                json.dump(frame_results, f, indent=2)


if __name__ == "__main__":
    main()
