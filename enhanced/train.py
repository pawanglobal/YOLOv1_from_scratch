import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Suppress TensorFlow info/warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Disable oneDNN optimizations for debugging or consistency

import torch                                    # For tensor operations and PyTorch functionalities
import torchvision.transforms as transforms     # For image transformations
import torch.optim as optim                     # For optimization algorithms
import torchvision.transforms.functional as FT  # Low-level image transformation functions
from torch.utils.tensorboard import SummaryWriter # A visualizer to monitor and understand model’s training process
from torch.utils.data import DataLoader         # For batching, shuffling, and loading data
import torch.onnx
from tqdm import tqdm                           # For creating progress bars during loops
from model import Yolov1                        # YOLOv1 model architecture
from dataset import VOCDataset                  # Custom dataset class for Pascal VOC data
from loss import YoloLoss                       # Custom loss function for YOLO
from utils import ( 
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

import random, time, datetime
from PIL import Image

torch.set_num_threads(12)
seed = 42                # Set a fixed seed for reproducibility -> ensures the same random numbers are generated on each run
torch.manual_seed(seed)  # Initialize PyTorch's random number generator

# 1. Hyperparameters and configuration settings

IMG_DIR = "data/images"                     # Directory containing image files
LABEL_DIR = "data/labels"                   # Directory containing label/annotation files
LOAD_MODEL_FILE = "vanilla_model.path.tar"  # File name for the saved model checkpoint
LOAD_MODEL = False                          # Whether to load a pre-trained model from a checkpoint

torch.set_num_threads(os.cpu_count())       # Use all logical CPU cores for PyTorch operations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
BATCH_SIZE = 16          # Number of images per batch -> 64 in the original paper but I don't have that much vram, grad accumulation
EPOCHS = 135             # Total number of training epochs (full passes through the dataset)
LEARNING_RATE = 2e-5     # Learning rate for optimizer 2e-5 = 2 * 10^(-5), original = 1e-3
WEIGHT_DECAY = 0.0005    # Regularization parameter to penalize large weights (helps prevent overfitting) -> Original: 0.0005
PATIENCE = 30            # To stop model early, if it is not learning anything, or anymore

NUM_WORKERS = 4                                   # Number of subprocesses for data loading (parallel loading)
PIN_MEMORY = True if DEVICE == "cuda" else False  # If True, DataLoader will copy tensors into CUDA pinned memory (no effect on CPU)

# 2. Data Augmentation

def clamp_boxes(boxes):
    """
    Ensure bounding box coordinates are within [0, 1] as per YOLOv1's normalized coordinate system
    """
    clamped = []
    for box in boxes:
        cls, x, y, w, h = box
        x = max(0, min(0.9999, x))
        y = max(0, min(0.9999, y))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        clamped.append([cls, x, y, w, h])
    return clamped

class Compose(object):
    """
    Custom Compose class to apply a sequence of transformations to both images and their corresponding bounding boxes.
    Built-in transform functions do not fully support joint transformations on images and bounding boxes together.
    """
    def __init__(self, transforms):
        self.transforms = transforms  # Store the list of transformation functions

    def __call__(self, img, bboxes):
        """
        Apply transformation t to the image and (ideally) bounding boxes.
        Here, we assume that each transformation is capable of handling both.
        If t doesn't support bounding box transformation, we need a custom implementation.
        Check if transform requires bboxes.
        """
        for t in self.transforms:
            if hasattr(t, 'requires_bbox') and t.requires_bbox:
                img, bboxes = t(img, bboxes)
            else:
                img = t(img)

        # Convert bboxes back to tensor, if necessary
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone().detach()  # Copy tensor
        elif len(bboxes) > 0:
            bboxes = torch.tensor(bboxes).clone().detach()
        else:
            bboxes = torch.zeros((0, 5))
        return img, bboxes

class RandomScaleTranslate:
    requires_bbox = True
    
    def __init__(self, max_shift=0.2, scale_range=(0.8, 1.2)):
        self.max_shift = max_shift
        self.scale_range = scale_range

    def __call__(self, img, bboxes):
        width, height = img.size
        
        # Random scaling
        scale = random.uniform(*self.scale_range)
        new_width, new_height = int(width * scale), int(height * scale)
        img = img.resize((new_width, new_height), Image.BILINEAR) # Image.BILINEAR for smoother resizing
        
        # Random translation
        max_dx = self.max_shift * width # Use original width for consistent shift
        max_dy = self.max_shift * height
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)
        
        # Create new image with translation
        new_img = Image.new(img.mode, (width, height), (0, 0, 0))
        paste_x = max(0, int(dx))  # Ensure non-negative paste position
        paste_y = max(0, int(dy))
        new_img.paste(img, (paste_x, paste_y))
        
        # Adjust bounding boxes
        new_bboxes = []
        for bbox in bboxes:
            # Unpack normalized bbox [class, x, y, w, h]
            cls, x, y, w, h = bbox
            
            # Convert to absolute coordinates
            x_abs = x * width
            y_abs = y * height
            w_abs = w * width
            h_abs = h * height
            
            # Apply scaling and translation
            x_abs = x_abs * scale + dx
            y_abs = y_abs * scale + dy
            w_abs *= scale
            h_abs *= scale
            
            # Convert back to normalized coordinates
            x_norm = max(0, min(1, x_abs / width))
            y_norm = max(0, min(1, y_abs / height))
            w_norm = max(0, min(1, w_abs / width))
            h_norm = max(0, min(1, h_abs / height))
            
            # Only keep boxes with valid dimensions
            if w_norm > 0.01 and h_norm > 0.01:
                new_bboxes.append([cls, x_norm, y_norm, w_norm, h_norm])
                
        return new_img, clamp_boxes(new_bboxes)

class RandomHorizontalFlip:
    requires_bbox = True
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            new_bboxes = []
            for bbox in bboxes:
                cls, x, y, w, h = bbox
                x = 1.0 - x  # Flip x-coordinate
                new_bboxes.append([cls, x, y, w, h])
            return img, clamp_boxes(new_bboxes)
        return img, bboxes
    
class RandomHSV:
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def __call__(self, img):
        # Convert to HSV
        img = img.convert('HSV')
        h, s, v = img.split()
        
        # Adjust saturation and value (exposure)
        sat_factor = random.uniform(1/self.factor, self.factor)
        val_factor = random.uniform(1/self.factor, self.factor)
        
        # clamp pixel values between 0 and 255, preventing overflow or underflow.
        s = s.point(lambda i: min(255, max(0, int(i * sat_factor))))
        v = v.point(lambda i: min(255, max(0, int(i * val_factor))))
        
        # Convert back to RGB
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        return img

train_transform = Compose([
    # RandomScaleTranslate(max_shift=0.2, scale_range=(0.8, 1.2)),
    # RandomHSV(factor=1.5),
    # RandomHorizontalFlip(p=0.5),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

test_transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# 3. Forward and Backward pass for one epoch

def train_fn(train_loader, model, optimizer, loss_fn, accum_steps=4):
    """
    Trains the model for one epoch using the provided data loader.
    Parameters:
        train_loader (DataLoader): Provides batches of training data (images and labels).
        model (nn.Module): The YOLOv1 model.
        optimizer (Optimizer): The optimizer for updating model weights.
        loss_fn (function): The loss function that calculates the error between predictions and targets.
        accum_steps: Batch size is compromised (limited computation power), experimenting with gradient accumulation 
                     to simulate a larger batch size. Example: accumulate gradients over 4 steps to approximate 
                     a batch size of BATCH_SIZE * 4
    """
    # Create a progress bar for the training loop; leave=True keeps the progress bar displayed after completion.
    loop = tqdm(train_loader, leave=True)
    mean_loss = []         # List to store loss for each batch
    
    accum_count = 0
    if accum_count ==0:
        optimizer.zero_grad()  # Clear gradients at the start
    
    pred_boxes = []
    target_boxes = []
    
    # Initialize component accumulators
    components_accum = {
        'coord': 0.0,
        'obj': 0.0,
        'noobj': 0.0,
        'class': 0.0
    }
    

    # Iterate over batches of data
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE) # Move the input data (x) and labels (y) to the device (CPU/GPU)
        out = model(x)                    # Get model predictions for the batch

        # Compute the loss between predictions and ground truth, then scale by 1/accum_steps to normalize gradients for accumulation
        loss, loss_components = loss_fn(out, y)
        scaled_loss = loss / accum_steps
        
        # Accumulate components
        for key in components_accum:
            components_accum[key] += loss_components[key].item()
        
        if batch_idx % 50 == 0:
            print(f" For Batch {batch_idx}: Scaled Loss (With Accumulated Gradients) = {scaled_loss.item():.4f}, Unscaled Loss = {loss.item():.4f}")

        # Debug: Print sample predicted and target boxes
        if batch_idx == 0: 
            with torch.no_grad():
                batch_pred_boxes = cellboxes_to_boxes(out)
                sample_pred = batch_pred_boxes[0][:2]  # First image, max 2 boxes
                sample_target = []
                label_matrix = y[0]  # Keep as tensor, shape: (7, 7, 30)
                for row in range(label_matrix.shape[0]):
                    for col in range(label_matrix.shape[1]):
                        if label_matrix[row, col, 20] == 1:  # Objectness score
                            class_label = int(torch.argmax(label_matrix[row, col, :20]))  # Tensor argmax
                            x_cell, y_cell, w_cell, h_cell = label_matrix[row, col, 21:25]
                            x_img = (col + x_cell) / 7.0
                            y_img = (row + y_cell) / 7.0
                            w_img = w_cell / 7.0
                            h_img = h_cell / 7.0
                            sample_target.append([class_label, 1.0, x_img, y_img, w_img, h_img])
                            if len(sample_target) == 2:  # Max 2 boxes
                                break
                        if len(sample_target) == 2:
                            break
                print(f"Sample Predicted Boxes (img 0): {sample_pred}")
                print(f"Sample Target Boxes (img 0): {sample_target}")

        mean_loss.append(loss.item())                # Append the numerical loss value for later averaging
        # optimizer.zero_grad()                      # Clear previous gradients -> Commented out to implement gradient accumulation
        scaled_loss.backward()                       # Backpropagate to compute new gradients
        accum_count += 1

        # Compute Training mAP (Diagnose Overfitting)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                batch_pred_boxes = cellboxes_to_boxes(out)
                for i in range(len(batch_pred_boxes)):
                    nms_boxes = non_max_suppression(
                        batch_pred_boxes[i], iou_threshold=0.5, threshold=0.25, box_format="midpoint"
                    )
                    pred_boxes.extend([[batch_idx * BATCH_SIZE + i, *box] for box in nms_boxes])

                    label_matrix = y[i]  # Keep as tensor, shape: (7, 7, 30)
                    for row in range(label_matrix.shape[0]):
                        for col in range(label_matrix.shape[1]):
                            if label_matrix[row, col, 20] == 1:  # Objectness score
                                class_label = int(torch.argmax(label_matrix[row, col, :20]))
                                x_cell, y_cell, w_cell, h_cell = label_matrix[row, col, 21:25]
                                x_img = (col + x_cell) / 7.0
                                y_img = (row + y_cell) / 7.0
                                w_img = w_cell / 7.0
                                h_img = h_cell / 7.0
                                target_boxes.append([batch_idx * BATCH_SIZE + i, class_label, 1.0, x_img, y_img, w_img, h_img])

        # Check if we’ve accumulated gradients for accum_steps batches (e.g., 4) or reached the last batch in the DataLoader.
        if accum_count == accum_steps or batch_idx == len(train_loader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0) # YOLOv1 didn't apply clipping, just for experiment purposes
            optimizer.step()       # Update model parameters based on the gradients
            optimizer.zero_grad()  # Clear gradients
            accum_count = 0

        # Update the progress bar with the current loss value
        loop.set_postfix(loss=loss.item())
    
    for key in components_accum:
        components_accum[key] /= len(train_loader)
    # Print the mean loss for the epoch
    mean_loss_scalar = sum(mean_loss) / len(mean_loss) if mean_loss else 0.0
    print(f"Mean loss was {mean_loss_scalar:.4f}")
    return mean_loss_scalar, pred_boxes, target_boxes, components_accum

# 4. Set the end-to-end training pipeline

def main():
    """
    Main workflow for setting up the model, datasets, loaders, and training the model
    """
    start_time = time.time()
    writer = SummaryWriter(log_dir="runs/vanilla_version")               # Initialize the TensorBoard
    
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE) # Initialize the YOLOv1 model
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)) 
    loss_fn = YoloLoss() # Initialize the YOLO loss function
    
    if LOAD_MODEL:
        """
        Load a pre-trained model checkpoint if LOAD_MODEL is True
        """
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Create the dataset using VOCDataset, which reads image paths and labels from a CSV file
    train_dataset = VOCDataset(
        "data/144examples.csv",
        transform=train_transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    validation_dataset = VOCDataset(
        "data/val.csv",
        transform=test_transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    
    test_dataset = VOCDataset(
        "data/test.csv",
        transform=test_transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    if len(train_dataset) < 100:
        print("WARNING: Small training dataset less than 100 may lead to overfitting. Consider using full PASCAL VOC dataset.")

    # Create a DataLoader for the dataset for batching, shuffling, and other parameters
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True      # drop test samples out of the batch size
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )
    
    # Validate Dataset (Rule Out Data Issues)
    if len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Validation or test dataset is empty. Check 'data/val.csv' and 'data/test.csv'.")
    
    for i in range(min(5, len(train_dataset))):
        img, labels = train_dataset[i]
        boxes = []
        for row in range(labels.shape[0]):
            for col in range(labels.shape[1]):
                if labels[row, col, 20] == 1:
                    class_label = torch.argmax(labels[row, col, :20]).item()
                    x_cell, y_cell, w_cell, h_cell = labels[row, col, 21:25]
                    x_img = (col + x_cell) / 7.0
                    y_img = (row + y_cell) / 7.0
                    w_img = w_cell / 7.0
                    h_img = h_cell / 7.0
                    boxes.append([class_label, x_img, y_img, w_img, h_img])
        boxes = torch.tensor(boxes) if boxes else torch.empty((0, 5))
        
        if boxes.size(0) == 0:
            print(f"Warning: Training sample {i} has no bounding boxes.")
        elif torch.any(boxes[:, 1:] < 0) or torch.any(boxes[:, 1:] > 1):
            print(f"Warning: Training sample {i} has invalid normalized coordinates: {boxes[torch.any(boxes[:, 1:] < 0, dim=1) | torch.any(boxes[:, 1:] > 1, dim=1)]}")
        elif torch.any(torch.isnan(img)) or torch.any(torch.isnan(boxes)):
            print(f"Warning: Training sample {i} contains NaN values.")
    
    patience = 0
    best_val_map = 0.0
    epoch_times = []

    for epoch in (range(EPOCHS)):
        epoch_start_time = time.time()
        loss_fn.epoch = epoch # Identifies if large losses come from specific components (e.g.,\sqrt{w} issues) or invalid predictions/targets

        # Visualization block (for debugging or inspection)
        if epoch % 10 == 0:
            with torch.no_grad():
                sample = next(iter(validation_loader))
                images, _ = sample
                images = images.to(DEVICE)
                predictions = model(images)
                for i in range(min(2, len(images))):
                    # Get cell-based bounding box predictions from the model and convert to image-level boxes
                    bboxes = cellboxes_to_boxes(predictions)
                    # Apply Non-Maximum Suppression (NMS) on the predicted boxes for the image at index 'i'
                    nms_boxes = non_max_suppression(
                        bboxes[i], 
                        iou_threshold=0.5,
                        threshold=0.25,     # Original: threshold = 0.25
                        box_format="midpoint"
                    )
                    # Plot the image (after permuting dimensions from (C, H, W) to (H, W, C)) with bounding boxes
                    plot_image(images[i].permute(1,2,0).cpu(), nms_boxes)

        # ===== LEARNING RATE SCHEDULE =====
        # Paper: "First raise from 10^-3 to 10^-2, then 10^-2 for 75 epochs,
        # then 10^-3 for 30 epochs, finally 10^-4 for 30 epochs"

        lr = LEARNING_RATE # For constant 'lr' just for experimentation purposes

        # if epoch <= 5:      # Warmup phase
        #     lr = LEARNING_RATE + (2e-4 - 2e-5) * (epoch / 5.0) # Original: (LEARNING_RATE) + (1e-2 - 1e-3) * (epoch / 5.0)
        # elif epoch <= 75:
        #     lr = 2e-4       # Original : 1e-2
        # elif epoch <= 105:
        #     lr = 2e-5       # Original: 1e-3
        # else:
        #     lr = 2e-6       # Original: 1e-4
        
        # Update optimizer LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Train the model for one epoch
        mean_loss, train_pred_boxes, train_target_boxes, loss_components = train_fn(train_loader, model, optimizer, loss_fn, accum_steps=4)
        
        # Log loss components to TensorBoard
        for key, value in loss_components.items():
            writer.add_scalar(f'Loss/{key}', value, epoch)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            try:
                train_map = mean_average_precision(
                train_pred_boxes, train_target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            except Exception as e:
                print(f"mAP calculation error: {e}")
                train_map = 0.0  # Handle any errors in mAP calculation
        else:
            # Use previous value for consistent logging
            train_map = train_map if 'train_map' in locals() else 0.0
        
        print(f"Train mAP: {train_map:.4f}")
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('Learning_Rate', lr, epoch)
        writer.add_scalar('mAP/train', train_map, epoch)
        
        # After computing train_map
        del train_pred_boxes, train_target_boxes
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Evaluate validation mAP for every 5 epochs, we can change it for experimental purposes
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred_boxes, target_boxes = get_bboxes(
                    validation_loader, model, iou_threshold=0.5, threshold=0.25 # Original: threshold=0.25
                ) 
                try:
                    val_map = mean_average_precision(
                        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
                    )
                except:
                    val_map = 0.0

            model.train()
            print(f"Epoch {epoch} | LR: {lr:.0e} | Validation mAP: {val_map:.4f} | Best mAP: {best_val_map:.4f}")
            writer.add_scalar('mAP/val', val_map, epoch)
            
            # Save model if validation mAP improves
            if val_map > best_val_map:
                best_val_map = val_map
                patience = 0 # reset patience counter
                checkpoint = {
                    "state_dict": model.state_dict(),       # Save model parameters
                    "optimizer": optimizer.state_dict(),    # Save optimizer state, we can avoid this to save more memory
                    "lr_schedule": {                        # Save current learning rate schedule
                        "epoch": epoch,
                        "best_val_map": best_val_map}
                }
                save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
                print(f"Saved model with Validation mAP: {val_map:.4f}")
                
            else:
                patience += 1

            # Overfitting detection
            if val_map > 0 and train_map > 0:
                overfit_ratio = min(train_map / max(val_map, 1e-06), 5.0) # Prevent divison by zero
                writer.add_scalar('Diagnostic/Overfit_Ratio', overfit_ratio, epoch)
                if overfit_ratio > 1.5:
                    print(f"Warning: Overfitting (train_map/val_map = {overfit_ratio:.1f})")

            # Stop if no improvement for 'PATIENCE' validations
            if val_map > 0 and patience >= PATIENCE:                        
                print(f"No improvement. Patience: {patience}/{PATIENCE}")
                print("Early stop triggered")
                break
        
        
                
        # Compute and print elapsed and remaining time
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        total_elapsed = time.time() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = EPOCHS - (epoch + 1)
        remaining_time = remaining_epochs * avg_epoch_time

        total_elapsed = str(datetime.timedelta(seconds=int(total_elapsed)))
        remaining_time = str(datetime.timedelta(seconds=int(remaining_time)))
        epoch_duration = str(datetime.timedelta(seconds=int(epoch_duration)))

        print(f"Epoch {epoch}: Elapsed Time={total_elapsed}, "
              f"Epoch Duration={epoch_duration}, "
              f"Estimated Remaining Time={remaining_time}")

    # Final test mAP
    model.eval()
    with torch.no_grad():
        # Evaluate model predictions on the training data
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.25 # originally threshold = 0.25 
        )
        # Compute mean Average Precision (mAP) for the predictions
        test_map = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        
    writer.add_scalar('mAP/test', test_map, EPOCHS)
    writer.close()
    print(f"Final Test mAP: {test_map}")
    
    # Visualize test predictions
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            for idx in range(min(8, len(x))):
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.25, box_format="midpoint")
                plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
            break

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = int(total_time % 60)
    print(f"Total training time: {hours}h {mins}m {secs}s")
    
    # Write to results file
    with open("training_results.txt", "w") as f:
        f.write(f"Final mAP: {test_map}\n")
        f.write(f"Total training time: {hours}h {mins}m {secs}s\n")
    
    # After training: Export trained model to ONNX format for inference and visualization
    dummy_input = torch.randn(1, 3, 448, 448).to(DEVICE)  
    try:
        torch.onnx.export(model, dummy_input, "model.onnx")     # Visualize with Netron
        print("Model exported to ONNX successfully")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Make sure you have onnx installed: pip install onnx")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
