import torch
from torch import nn

import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

import numpy as np
from functools import partial

class FCModel(nn.Module):
    """
    Part 1.a: Fully connected neural network
    """

    def __init__(self):
        super(FCModel, self).__init__()
        # TODO: Define the layers for the fully connected neural network
        # Use nn.Flatten, nn.Linear, and nn.ReLU appropriately
        # expected architecture:
        # flatten -> Linear(c=256) -> ReLU -> Linear(c=256) -> ReLU -> Linear
        self.arch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.arch(x)
        return logits

class CNNModel(nn.Module):
    """
    Part 1.b: Convolutional neural network
    """

    def __init__(self):
        super(CNNModel, self).__init__()
        # TODO: Define the convolutional and fully connected layers for the CNN
        # Use nn.Conv2d, nn.MaxPool2d, nn.Linear, Flatten, and nn.ReLU appropriately
        # expected architecture:
        # Conv(c=32, s=1) -> ReLU -> MaxPool(s=2) -> Conv(c=64, s=1) -> ReLU -> flatten -> Linear
        self.arch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6400, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.arch(x)
        return logits


class ClassificationLoss(nn.Module):
    """
    Part 2: Loss function

    Implement softmax cross entropy loss without using torch.nn
    """

    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def calculate_loss(self, predicted, target):
        one_hot = torch.zeros(target.size(0), predicted.shape[1])
        one_hot[torch.arange(target.size(0)), target] = 1
        
        exp = torch.exp(predicted)
        soft = exp / torch.sum(exp)
        
        dot_product = one_hot * torch.log(soft)
        return -torch.sum(dot_product)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.calculate_loss(y_pred, y_true)


class BetterModel(nn.Module):
    """
    Part 3: Better Model: ResNet
    """
    
    def create_model(self, num_classes=91):
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        return model

    def __init__(self):
        super(BetterModel, self).__init__()
        self.arch = self.create_model(1)

    def forward(self, x: torch.Tensor, targets=None) -> torch.Tensor:
        return self.arch(x, targets)