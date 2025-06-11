from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image

import os
import torch
import json

def SFCHD_transforms(image, target):
    output_size = (1280, 704)
    # Define any target transformations here
    class_ids = []
    bboxes = []
    for line in target:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip invalid lines
        class_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:5])
        # Convert to absolute pixel values
        orig_w, orig_h = image.size
        x_c *= orig_w
        y_c *= orig_h
        w *= orig_w
        h *= orig_h

        bboxes.append([x_c, y_c, w, h])
        class_ids.append(class_id)

    # Resize image
    image = image.resize(output_size, resample=Image.BILINEAR)
    scale_x = output_size[0] / orig_w
    scale_y = output_size[1] / orig_h

    # Resize bboxes and normalize to [0, 1] in cxcywh format
    bboxes_resized = []
    for box in bboxes:
        x_c, y_c, w, h = box
        x_c = x_c * scale_x
        w = w * scale_x
        y_c = y_c * scale_y
        h = h * scale_y
        # Normalize
        x_c /= output_size[0]
        w /= output_size[0]
        y_c /= output_size[1]
        h /= output_size[1]
        bboxes_resized.append([x_c, y_c, w, h])

    target = {
        'labels': torch.tensor(class_ids, dtype=torch.int64),
        'boxes': torch.tensor(bboxes_resized, dtype=torch.float32)
    }
    return transforms.ToTensor()(image), target

def SFCHD_transform(image):
    # Define any image transformations here
    return image

def SFCHD_target_transform(target):
    # Define any target transformations here
    return target

class SFCHD(VisionDataset):
    def __init__(self, root, image_folder, transform=None, target_transform=None, transforms=SFCHD_transforms, train=True):
        super(SFCHD, self).__init__(root, transforms=transforms,
                                    transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train
        self.labels_folder = 'train' if self.train else 'val'
        assert os.path.exists(self.root), f"Dataset root directory {self.root} does not exist."

        file_name = 'train.txt' if train else 'val.txt'
        with open(os.path.join(self.root, file_name), 'r') as f:
            self.file_names = f.readlines()

        # Get all files in the dataset
        self.file_names = sorted([line.strip() for line in self.file_names])
        self.image_folder = image_folder
        assert len(self.file_names) > 0, "No files found in the dataset."
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # Implement the logic to load an item from the dataset
        image_file_name = self.file_names[index]
        file_name = image_file_name.split('/')[-1].replace('.jpg', '.txt') if image_file_name.endswith('.jpg') else image_file_name.replace('.png', '.txt')
        image_path = os.path.join(self.image_folder, image_file_name)

        # Load the image and apply transformations
        image = Image.open(image_path).convert('RGB')
        with open(os.path.join(self.root, self.labels_folder, file_name), 'r') as f:
            target = f.readlines()
        image, target = self.transforms(image, target) if self.transforms else (image, target)
        if self.transform is not None:
            image = self.transform(image)  
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        # Implement the logic to return the length of the dataset
        return len(self.file_names)
    
def collate_fn(batch):
    """
    Custom collate function to handle variable-sized images.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Return images and targets as a tuple

def CPPE_transforms(image, target):
    new_target = []
    class_ids = []
    boxes = []
    for each in target:
        class_ids.append(each['category_id'])
        boxes.append(each['bbox'])
    # Convert bounding boxes to cxcywh format
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        x_c = x + w / 2
        y_c = y + h / 2
        boxes[i] = [x_c, y_c, w, h]
    # Resize image
    output_size = (1280, 704)
    image = image.resize(output_size, resample=Image.BILINEAR)
    scale_x = output_size[0] / image.width
    scale_y = output_size[1] / image.height
    # Resize and normalize bounding boxes  
    for i in range(len(boxes)):
        x_c, y_c, w, h = boxes[i]
        x_c = x_c * scale_x / output_size[0]
        y_c = y_c * scale_y / output_size[1]
        w = w * scale_x / output_size[0]
        h = h * scale_y / output_size[1]
        boxes[i] = [x_c, y_c, w, h]
    new_target = {
        'labels': torch.tensor(class_ids, dtype=torch.int64),
        'boxes': torch.tensor(boxes, dtype=torch.float32)
    }
    return transforms.ToTensor()(image), new_target

class CPPEDataset(VisionDataset):
    def __init__(self, root, image_folder, transform=None, target_transform=None, train=True):
        super(CPPEDataset, self).__init__(root, transforms=None,
                                          transform=transform, target_transform=target_transform)
        self.root = root
        self.image_folder = image_folder
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # Load file names from the dataset
        temp_annotation_file = os.path.join(self.root, 'train.json' if self.train else 'test.json')
        with open(temp_annotation_file, 'r') as f:
            annotations = json.load(f)
            self.image_list = annotations['images']
            self.annotation = annotations['annotations']

    def __getitem__(self, index):
        image_file_name = os.path.join(self.image_folder, self.image_list[index])
        annotation = self.annotation[index] 

        # Load the image and apply transformations
        image = Image.open(image_file_name).convert('RGB')
        image, target = self.transforms(image, annotation) if self.transforms else (image, annotation)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return image, target

    def __len__(self):
        return len(self.image_list)