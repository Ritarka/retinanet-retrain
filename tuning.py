import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial
import os
import matplotlib.pyplot as plt

from data import get_dataloaders
import neural_network
import trainer
import visualize
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import argparse


plt.rcParams['font.family'] = 'DejaVu Sans'
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.memory_summary(device=None, abbreviated=False)
# Set up distributed training
dist.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.getenv("LOCAL_RANK", 0))  # Use LOCAL_RANK from distributed environment
device = f'cuda:{local_rank}'
torch.cuda.set_device(local_rank)


# Hyperparameters
batch_size = 4
learning_rate = 0.01
epochs = 20

# Data transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Get data loaders with DistributedSampler
trainloader, testloader = get_dataloaders(
    dataset_name="imagenet-tiny",
    batch_size=batch_size,
    train_transforms=transform_train,
    test_transforms=transform_test,
)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# Initialize the model
model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=None
            # detections_per_img=5,
            # fg_iou_thresh = 0.8,
            # bg_iou_thresh = 0.5
        )
# num_anchors = model.head.classification_head.num_anchors
# model.head.classification_head = RetinaNetClassificationHead(
#     in_channels=256,
#     num_anchors=num_anchors,
#     num_classes=2,
#     norm_layer=partial(torch.nn.GroupNorm, 32)
# )

# Freeze the backbone
# for param in model.backbone.parameters():
#     param.requires_grad = False

# Move model to device before wrapping it in DDP
model = model.to(device)

# Wrap the model in DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, 
                                                  #find_unused_parameters=True, 
                                                  )

# Use RetinaNet's internal loss functions
params_to_optimize = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


criterion = MeanAveragePrecision
# criterion = nn.CrossEntropyLoss()

# Training loop
train_losses, test_losses = trainer.train(model, optimizer, criterion, trainloader, testloader, epochs, device)

# Evaluate and visualize results
# accuracy = trainer.evaluate_accuracy(model, testloader, device)
# print("Test Accuracy:", accuracy)
# visualize.visualize_loss(train_losses, test_losses, "Best Model")
# visualize.visualize_images(model, testloader.dataset, device, "Best Model")
