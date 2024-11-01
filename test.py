from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch.cuda
import torchvision
import shutil
from functools import partial
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

from rich import print

from torch.utils.data import DataLoader

from dataset import *

import torchvision.transforms as transforms

from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def draw_bbox(img, retinanet_result, color="#0000ff"):
    # Extract the bounding box coordinates from the RetinaNet result
    boxes = retinanet_result['boxes'].cpu().detach().numpy()  # Convert tensor to NumPy array
    # scores = retinanet_result['scores'].cpu().detach().numpy()  # Convert tensor to NumPy array
    # labels = retinanet_result['labels'].cpu().detach().numpy()  # Convert tensor to NumPy array
    
    for i, box in enumerate(boxes):
        # score = scores[i]
        # label = labels[i]  # This would be the class index; you can map it to a name if needed
        
        # Extract the box coordinates
        x_min, y_min, x_max, y_max = box
        # print(x_min, y_min, x_max, y_max)
        
        
        d = ImageDraw.Draw(img)
        d.rectangle([int(x_min), int(y_min), int(x_max), int(y_max)], outline=color, width=1)
        
        # color = (0, 255, 0)  # Green bounding box
        # thickness = 3
        # cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
    
    return img


def main():   
    # paths = ["videos/easy/video0_cam0", "videos/easy/video1_cam0", 
    #          "videos/hard/video0_cam0", "videos/hard/video2_cam0",
    #          "videos/hard/video3_cam0", "videos/hard/video4_cam0",
    #          "videos/hard/video5_cam0", "videos/hard/video6_cam0"]
    
    # paths = ["videos/038000138720/hard/video0_cam0", "videos/080000514202/hard/video0_cam0"]
    paths = ["/home/biometrics/ritarka/vision/data/tiny-imagenet-200"]
    
    
    for path in paths:
        print(path)
        detect(Path(path), crop=False)  # Set crop=True to use cropping

        
def move_to_cuda(data, device):
    if isinstance(data, dict):
        return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cuda(item, device) for item in data]
    else:
        return data.to(device)


def detect(path: Path, crop=True):  # Add a flag to switch between cropping and drawing a bounding box
    
    if not path.exists():
        raise Exception("Invalid filepath to " + str(path))
    
    transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    test_dataset = TinyImageNet(path / "val", transform=transform_test, validation=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    
    shutil.rmtree(path / "cropped", ignore_errors=True)
    
    
    detector = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        # detections_per_img=1,
        # # topk_candidates=5000,
        # # fg_iou_thresh=0.9,
        # # nms_thresh=0.75,
        score_thresh=0.5
    )

    checkpoint = torch.load("/home/biometrics/ritarka/vision/train/checkpoints/model_checkpoint_epoch_last.pth"
                            , map_location=torch.device('cpu'))
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v

    detector.load_state_dict(new_state_dict)

    detector = detector.to(device)
    detector.eval()

    (path / "cropped").mkdir(parents=True, exist_ok=True)
    
    map = MeanAveragePrecision(box_format='xyxy', iou_type="bbox", extended_summary=True)
    detector.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            inputs, targets = data

            # Load inputs and labels to device
            inputs = inputs.to(device, non_blocking=True)
            targets = [{'boxes': targets['boxes'][i], 'labels': targets['labels'][i]} for i in range(len(targets['boxes']))]
            targets = move_to_cuda(targets, device)
            
            # Use autocast for mixed precision
            result = detector(inputs)
            map.update(result, targets)
                            
            image = transforms.functional.to_pil_image(inputs[0])
            image = draw_bbox(image, result[0], color="#00ff00")
            image = draw_bbox(image, targets[0], color="#ff0000")
            plt.imshow(image)
            plt.savefig(path / "cropped" / f"image_{i}.png")
            
    print(f"Got mAP of {map.compute()}")
            

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (device == "cpu"): exit()
    
    main()