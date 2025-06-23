import torch
import torch.nn as nn
from utils import intersection_over_union  # Utility function to calculate IoU

class YoloLoss(nn.Module):
    """
    Loss Function: To calculate the localization, confidence and the class loss
    Inherit from nn.Module to integrate with PyTorch
    """
    def __init__(self, S=7, B=2, C=20):
        """
        S = Grid size (number of cells in one dimension, i.e., S x S grid)
        B = Number of bounding boxes predicted per grid cell
        C = Number of classes (e.g., 20 for Pascal VOC)
        """
        super(YoloLoss, self).__init__()        # Calls the parent class constructor
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum") # 'reduction="sum"' means that errors are summed over all elements.
        self.S = S
        self.B = B
        self.C = C

        # Loss scaling factors:
        self.lambda_noobj = 0.5 # lambda_noobj scales loss from grid cells with no object (to reduce their influence) -> Original: self.lambda_noobj = 0.5
        self.lambda_coord = 5 # lambda_coord scales the loss from bounding box coordinate errors -> more emphasis on accurate localization.
        # self.lambda_obj = 2.0 # For experiment, not mentioned in the orginal paper
        # self.lambda_class = 2.0 # Only for experiment purposes, not mentioned in the orginal paper

    def forward(self, predictions, target):
        """
        Reshape predictions to shape: (batch_size, S, S, (C + B*5))
        Calculate Intersection over Union (IoU) for both predicted bounding boxes:
        predictions[..., 21:25] corresponds to box 1 coordinates (x, y, w, h)
        predictions[..., 26:30] corresponds to box 2 coordinates (x, y, w, h)
        target[..., 21:25] holds the ground truth box coordinates.
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5) # The -1 automatically infers the batch size.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        # Concatenate IoU values for the two boxes along a new dimension.
        # unsqueeze(0) adds an extra dimension so that we have a tensor of shape (2, ...).
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # For each grid cell, select the box with the maximum IoU.
        # iou_maxes: the best IoU value, best_box: index (0 or 1) of the best box.
        iou_maxes, best_box = torch.max(ious, dim=0)
        
        # Extract the objectness indicator from the target -> target[..., 20] is the objectness flag (1 if object exists, 0 otherwise).
        # unsqueeze(3) adds an extra dimension for broadcasting.
        exists_box = target[..., 20].unsqueeze(3)

        # ===================================================== #
        #               BOX COORDINATES LOSS                    #
        # ===================================================== #

        # Select the predicted box (from box1 or box2) based on best_box indicator:
        box_predictions = exists_box * (
            best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25]
        )

        # Ground truth bounding boxes, only for cells with objects
        box_targets = exists_box * target[..., 21:25]

        # Apply square root transformation to width and height -> this transformation reduces the impact of large errors on bigger boxes
        box_predictions[..., 2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # Compute loss for bounding box coordinates. Flatten the tensors so that they have shape (N*S*S, 4) for MSE computation. 
        # Where N -> Batch Size and 4 is coordinates.
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ===================================================== #
        #                OBJECTNESS LOSS                        #
        # ===================================================== #
        
        # Select the objectness score from the best box.
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        
        # Compute loss for objectness: Only consider grid cells that contain objects.
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ===================================================== #
        #                 NO OBJECT LOSS                        #
        # ===================================================== #
        
        # Compute loss for cells without an object. Flatten from dimension 1 to combine S*S cells.
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        
        # Also add loss for the second predicted box (even for cells without objects).
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ===================================================== #
        #                  CLASS LOSS                           #
        # ===================================================== #
        
        # Compute loss for class predictions. Only compute for grid cells with objects (exists_box mask).
        class_loss = self.cross_entropy(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        # ===================================================== #
        #           COMBINE ALL LOSS COMPONENTS                 #
        # ===================================================== #
        
        loss = (
            self.lambda_coord * box_loss            # Weighted coordinate loss
            + object_loss                           # Objectness loss
            + self.lambda_noobj * no_object_loss    # Weighted no-object loss
            + class_loss                            # Class prediction loss
        )

        # Return components dictionary
        loss_components = {
            'coord': box_loss.detach(),
            'obj': object_loss.detach(),
            'noobj': no_object_loss.detach(),
            'class': class_loss.detach()
        }

        # Debug print statement for loss
        # print(f" Epoch {self.epoch}: coord_loss={(self.lambda_coord * box_loss).item():.4f}, obj_loss={(object_loss).item():.4f}, "
        #     f"noobj_loss={(self.lambda_noobj * no_object_loss).item():.4f}, class_loss={class_loss.item():.4f}, total_loss={loss.item():.4f}")
        # if loss.item() > 1e4:  # Flag large losses
        #     print("Warning: Large loss detected. Check predictions and targets.")
        #     print(f"Predictions min/max: {predictions.min().item():.4f}/{predictions.max().item():.4f}")
        #     print(f"Target min/max: {target.min().item():.4f}/{target.max().item():.4f}")

        return loss, loss_components  # Return the combined loss
