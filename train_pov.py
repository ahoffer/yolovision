import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)


class COCODataset(Dataset):
    """Dataset class for COCO object detection."""

    def __init__(self, root_dir: str, annotation_file: str, transform: Optional[transforms.Compose] = None,
                 validation: bool = False, val_split: float = 0.2):
        """
        Args:
            root_dir: Root directory containing the dataset
            annotation_file: Path to COCO annotation JSON file
            transform: Optional transforms to apply to images
            validation: Whether this is a validation dataset
            val_split: Fraction of data to use for validation
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.validation = validation

        # Load COCO annotations
        logger.info(f"Loading COCO annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)

        # Create category ID to continuous index mapping
        self.cat_id_to_idx = {cat['id']: idx + 1  # Add 1 because 0 is background
            for idx, cat in enumerate(self.coco['categories'])}

        # Store category names for reference
        self.cat_names = {cat['id']: cat['name'] for cat in self.coco['categories']}

        # Create image ID to filename mapping
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.coco['images']}

        # Group annotations by image
        self.image_annotations = self._group_annotations()

        # Split dataset
        all_image_ids = list(self.image_annotations.keys())
        np.random.seed(42)  # for reproducibility
        np.random.shuffle(all_image_ids)

        split_idx = int(len(all_image_ids) * val_split)
        self.image_ids = all_image_ids[:split_idx] if validation else all_image_ids[split_idx:]

        logger.info(f"Dataset split: {'validation' if validation else 'training'} "
                    f"with {len(self.image_ids)} images")

        # Calculate class weights
        self.class_weights = self._calculate_class_weights()

    def _group_annotations(self) -> Dict:
        """Groups annotations by image ID."""
        grouped = {}
        for ann in self.coco['annotations']:
            image_id = ann['image_id']
            if image_id not in grouped:
                grouped[image_id] = []
            grouped[image_id].append(ann)
        return grouped

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculates class weights for balanced sampling."""
        label_counts = {}
        for anns in self.image_annotations.values():
            for ann in anns:
                cat_idx = self.cat_id_to_idx[ann['category_id']]
                label_counts[cat_idx] = label_counts.get(cat_idx, 0) + 1

        max_count = max(label_counts.values())
        weights = {label: max_count / count for label, count in label_counts.items()}
        return torch.tensor([weights.get(i, 1.0) for i in range(len(self.cat_id_to_idx) + 1)])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]

        # Load image
        img_filename = self.image_id_to_file[image_id]
        img_path = self.root_dir / img_filename
        image = Image.open(img_path).convert('RGB')

        # Get annotations for this image
        anns = self.image_annotations.get(image_id, [])

        # Convert COCO format to expected format
        boxes = []
        labels = []
        areas = []  # Added for RetinaNet v2
        iscrowd = []  # Added for RetinaNet v2

        for ann in anns:
            # COCO bbox format is [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

        # Handle images with no annotations
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return additional information needed by RetinaNet v2
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([image_id]), 'area': areas,
            'iscrowd': iscrowd}

        return image, target


class COCOTrainer:
    """Handles the training process for COCO object detection using RetinaNet v2."""

    def __init__(self, dataset_path: str, annotation_file: str, device: torch.device):
        self.dataset_path = dataset_path
        self.device = device

        # Create datasets with augmentations
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.train_dataset = COCODataset(dataset_path, annotation_file, transform=train_transform, validation=False)

        self.val_dataset = COCODataset(dataset_path, annotation_file, transform=val_transform, validation=True)

        # COCO has 80 categories + background
        num_classes = 81

        # Initialize RetinaNet v2 model
        self.model = retinanet_resnet50_fpn_v2(weights_backbone="IMAGENET1K_V2",  # Use improved backbone weights
            num_classes=num_classes, trainable_backbone_layers=5  # Train all backbone layers
        )

        # Enable AMP for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        self.model.to(device)

    def train(self, num_epochs: int, batch_size: int, learning_rate: float = 0.0001) -> nn.Module:
        """Trains the model with mixed precision."""
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn,
            num_workers=4, pin_memory=True  # Faster data transfer to GPU
        )

        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn,
            num_workers=4, pin_memory=True)

        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=learning_rate,
            weight_decay=0.0001, amsgrad=True  # Use AMSGrad variant
        )

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)

            # Validation phase
            val_loss = self._validate(val_loader)

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")

            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self.model

    def _train_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Trains for one epoch using mixed precision."""
        self.model.train()
        total_loss = 0

        for images, targets in data_loader:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += losses.item()

        return total_loss / len(data_loader)

    def _validate(self, data_loader: DataLoader) -> float:
        """Performs validation."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

        return total_loss / len(data_loader)

    @staticmethod
    def _collate_fn(batch: List) -> Tuple:
        """Custom collate function for the data loader."""
        return tuple(zip(*batch))

    def save_model(self, path: str) -> None:
        """Saves the model and training state."""
        torch.save({'model_state_dict': self.model.state_dict(), 'category_mapping': self.train_dataset.cat_id_to_idx,
            'category_names': self.train_dataset.cat_names}, path)
        logger.info(f"Model saved to {path}")


def main():
    # Configuration
    DATASET_PATH = "./coco/train2017"  # Path to COCO images
    ANNOTATION_FILE = "./coco/annotations/instances_train2017.json"  # Path to COCO annotations
    NUM_EPOCHS = 10
    BATCH_SIZE = 4

    try:
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Initialize trainer and train model
        trainer = COCOTrainer(DATASET_PATH, ANNOTATION_FILE, device)
        model = trainer.train(NUM_EPOCHS, BATCH_SIZE)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
