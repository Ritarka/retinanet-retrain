import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split, Dataset

class TinyImageNet(Dataset):
    image_paths = []
    boxes = []
    
    transform = None
    
    def __init__(self, root_dir, transform=None, validation=False):
        """
        Custom Dataset to load images and boxes from multiple directories.

        :param root_dir: List of directories that contain "images" and "boxes" subdirectories.
        :param transform: Optional transforms to apply to the images.
        """
        
        self.transform = transform
        
        subfolders = os.listdir(root_dir)
        for folder in subfolders:
            path = root_dir / folder / (f"{str(folder)}_boxes.txt")
            img_folder_path = root_dir / folder / "images"
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if not validation:
                        name, x1, y1, x2, y2, = line.split()
                    else:
                        name, something, x1, y1, x2, y2, = line.split()

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    img_path = img_folder_path / name
                    
                    def bound(x, a, b):
                        if x < a:
                            x = a
                        elif x > b:
                            x = b
                        return x
                    
                    x1 = bound(x1, 0, 64)
                    x2 = bound(x2, 0, 64)
                    y1 = bound(y1, 0, 64)
                    y2 = bound(y2, 0, 64)
                    
                    if x2 - x1 <= 0: continue
                    if y2 - y1 <= 0: continue
                    
                    self.image_paths.append(img_path)
                    self.boxes.append([x1, y1, x2, y2])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed

        box = self.boxes[idx]        
        box = torch.tensor(box, dtype=torch.int64)

        image = self.transform(image)
        c, h, w = image.shape

        # print(c, h, w)

        target = {
            'boxes': torch.reshape(box, (1, 4)),  # Shape: (N, 4) for N boxes
            'labels': torch.reshape(torch.tensor(0), (1,))  # Shape: (N,)
        }

        return image, target


class ImageLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset to load images and labels from multiple directories.

        :param root_dir: List of directories that contain "images" and "labels" subdirectories.
        :param transform: Optional transforms to apply to the images.
        """
        self.image_paths = []
        self.label_paths = []
        self.transform = transform

        # Collect all image and label file paths
        for root_dir in root_dir:
            image_dir = os.path.join(root_dir, 'images')
            label_dir = os.path.join(root_dir, 'labels')

            for img_file in os.listdir(image_dir):
                img_path = os.path.join(image_dir, img_file)
                label_file = img_file.replace('.jpg', '.txt')  # Assuming label files are .txt; adjust if needed
                label_path = os.path.join(label_dir, label_file)

                # Check if both image and label exist
                if os.path.exists(img_path) and os.path.exists(label_path):
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed

        # Load label
        label_path = self.label_paths[idx]
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                # Example format: "0 x1 y1 x2 y2 x3 y3 x4 y4"
                label_split = line.strip().split()
                class_number = int(label_split[0])  # Class number (first value)
                obb_coords = torch.tensor([float(coord) for coord in label_split[1:]])  # Extract the 8 coordinates

                # Convert OBB to AABB
                aabb = obb_to_aabb(obb_coords)  # Function we defined earlier
                boxes.append(aabb)
                labels.append(class_number)

        # Convert to tensors
        boxes = torch.stack(boxes)  # Shape: (N, 4) for N boxes
        labels = torch.tensor(labels, dtype=torch.int64)  # Shape: (N,)

        # Apply transformations to the image if necessary
        image = self.transform(image)
        c, h, w = image.shape

        # Adjust AABB coordinates to the image size
        boxes[:, [0, 2]] *= w  # Scale x coordinates (x_min, x_max)
        boxes[:, [1, 3]] *= h  # Scale y coordinates (y_min, y_max)

        # Prepare the target in the format the model expects:
        # - boxes (FloatTensor[N, 4]): ground-truth boxes in [x1, y1, x2, y2]
        # - labels (Int64Tensor[N]): the class label for each ground-truth box
        target = {
            'boxes': boxes,  # Shape: (N, 4) for N boxes
            'labels': labels  # Shape: (N,)
        }

        return image, target



def obb_to_aabb(obb):
    """
    Converts an Oriented Bounding Box (OBB) defined by 4 points into
    an Axis-Aligned Bounding Box (AABB).

    :param obb: Tensor of shape (8,) representing 4 corners of the OBB 
                in the format [x1, y1, x2, y2, x3, y3, x4, y4].
    :return: Tensor of shape (4,) representing the AABB in the format 
             [x_min, y_min, x_max, y_max].
    """
    # Extract x and y coordinates
    x_coords = obb[0::2]  # [x1, x2, x3, x4]
    y_coords = obb[1::2]  # [y1, y2, y3, y4]

    # Compute the minimum and maximum coordinates
    x_min = torch.min(x_coords)
    y_min = torch.min(y_coords)
    x_max = torch.max(x_coords)
    y_max = torch.max(y_coords)

    return torch.tensor([x_min, y_min, x_max, y_max])
