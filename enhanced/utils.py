import torch
import numpy as np                      # For mathematics
import matplotlib.pyplot as plt         # For plotting
import matplotlib.patches as patches    # For plotting
from collections import Counter         # For tallying; counting the frequency of an element

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates Intersection over Union (IoU) between predicted and ground truth bounding boxes.

    Parameters:
        boxes_preds (tensor): Predicted bounding boxes of shape (BATCH_SIZE, 4). 
            If box_format == "midpoint", boxes are in the format (x_center, y_center, width, height).
            If box_format == "corners", boxes are in the format (x1, y1, x2, y2).
        boxes_labels (tensor): Ground truth bounding boxes with the same shape and format.
        box_format (str): Indicates the format of the bounding boxes ("midpoint" or "corners").

    Returns:
        tensor: IoU for each pair of predicted and ground truth boxes.
    """
    
    # If boxes are in "midpoint" format: (x_center, y_center, width, height)
    if box_format == "midpoint":
        # Convert predicted boxes to corner format (x1, y1, x2, y2)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2  # left coordinate
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2  # top coordinate
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2  # right coordinate
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2  # bottom coordinate

        # Convert ground truth boxes to corner format
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # If boxes are already in "corners" format: (x1, y1, x2, y2)
    if box_format == "corners":
        # Use the coordinates as provided for predicted boxes
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        # And for ground truth boxes
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Calculate the coordinates of the intersection rectangle
    x1 = torch.max(box1_x1, box2_x1)  # max of left coordinates
    y1 = torch.max(box1_y1, box2_y1)  # max of top coordinates
    x2 = torch.min(box1_x2, box2_x2)  # min of right coordinates
    y2 = torch.min(box1_y2, box2_y2)  # min of bottom coordinates

    # Compute the area of intersection
    # .clamp(0) ensures that we don't get negative widths or heights if boxes don't overlap
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Compute the area of each box
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Compute IoU:
    # Intersection over Union = Intersection / (Area of Box1 + Area of Box2 - Intersection)
    # A small constant (1e-6) is added to the denominator for numerical stability.
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Performs Non-Maximum Suppression (NMS) on a list of bounding boxes.

    Parameters:
        bboxes (list): A list of bounding boxes, where each bounding box is specified as:
                       [class_pred, prob_score, x1, y1, x2, y2].
                       - class_pred: The predicted class label.
                       - prob_score: The confidence score of the prediction.
                       - (x1, y1, x2, y2): The coordinates of the bounding box.
        iou_threshold (float): IoU threshold to decide whether boxes overlap too much.
                                 If the IoU between two boxes is higher than this threshold,
                                 one of them will be suppressed.
        threshold (float): Confidence threshold. Boxes with a confidence score lower than
                           this value will be removed before NMS is applied.
        box_format (str): The format of the bounding boxes. Can be "midpoint" (center, width, height)
                          or "corners" (x1, y1, x2, y2).

    Returns:
        list: The list of bounding boxes after applying Non-Maximum Suppression.
    """

    # Ensure that the input is a list of bounding boxes
    assert type(bboxes) == list

    # Filter out all bounding boxes that have a confidence score less than the threshold.
    # Each box is a list, and box[1] holds the confidence score.
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort the remaining bounding boxes by their confidence score in descending order.
    # This means the box with the highest confidence is processed first.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # This will store the final bounding boxes after applying NMS.
    bboxes_after_nms = []

    # Process the list until there are no more boxes to compare.
    while bboxes:
        # Select the bounding box with the highest confidence score (first in the sorted list)
        chosen_box = bboxes.pop(0)

        # For each of the remaining boxes, only keep it if:
        # - It is of a different class than the chosen_box, OR
        # - Its Intersection over Union (IoU) with the chosen_box is less than the threshold.
        # This step effectively suppresses boxes that overlap too much with a high-confidence box.
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # Different predicted class, or...
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),  # Convert chosen_box coordinates to a tensor
                torch.tensor(box[2:]),         # Convert current box coordinates to a tensor
                box_format=box_format,         # Use the specified format ("midpoint" or "corners")
            ) < iou_threshold                # Only keep if IoU is less than the threshold
        ]

        # Append the chosen_box (the one with the highest confidence) to the final list.
        bboxes_after_nms.append(chosen_box)

    # Return the final list of bounding boxes after suppression.
    return bboxes_after_nms




def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device = DEVICE
    # device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes




def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from YOLO (which are relative to each grid cell)
    into bounding boxes relative to the entire image. Model ouputs predictions for each grid cell.

    Why? So we can easily visualize and use the bounding boxes.
    
    The predictions are assumed to be arranged for an SxS grid with each cell predicting
    30 values (for YOLOv1: 20 class scores + 2 * 5 bounding box parameters).
    
    Parameters:
        predictions (tensor): Model output tensor.
        S (int): The grid size (default is 7 for a 7x7 grid).
        
    Returns:
        tensor: Converted predictions with bounding boxes in the format:
            [predicted_class, best_confidence, x_center, y_center, width, height]
            where coordinates are relative to the entire image (range 0 to 1).
    """
    
    # Move predictions to CPU if they are not already
    predictions = predictions.to("cpu")
    
    # Get the batch size from the first dimension of predictions
    batch_size = predictions.shape[0]
    
    # Reshape predictions to (batch_size, S, S, 30)
    # Each grid cell now has a vector of 30 values.
    predictions = predictions.reshape(batch_size, S, S, 30)
    
    # Extract the bounding boxes parameters predicted by each of the two boxes per cell.
    # bboxes1: parameters from the first box, indices 21 to 24 (inclusive of 21 and exclusive of 25)
    bboxes1 = predictions[..., 21:25]
    # bboxes2: parameters from the second box, indices 26 to 29
    bboxes2 = predictions[..., 26:30]
    
    # Extract the confidence scores for both boxes.
    # predictions[..., 20] corresponds to the confidence of the first box.
    # predictions[..., 25] corresponds to the confidence of the second box.
    # unsqueeze(0) adds a new dimension so that we can concatenate along the new axis.
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    
    # Determine which of the two boxes (for each cell) has the highest confidence.
    # best_box will be 0 if the first box is better, or 1 if the second box is better.
    best_box = scores.argmax(0).unsqueeze(-1)  # add an extra dimension at the end
    
    # Select the best bounding box for each grid cell:
    # If best_box is 0, then (1 - best_box) will be 1 and best_box will be 0,
    # meaning bboxes1 is chosen. If best_box is 1, then bboxes2 is chosen.
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    # Create an index tensor for grid cell positions along the x-axis.
    # torch.arange(7) creates values from 0 to 6. We repeat this for each batch and each row.
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    
    # Adjust the bounding box coordinates relative to the whole image:
    # The network outputs coordinates relative to the cell. To get image-relative values:
    # - Add the cell indices to the predicted center coordinates.
    # - Then divide by S so that the values range from 0 to 1.
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    # For y, we need to permute the cell indices to align with the grid's second dimension.
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    # The width and height are also scaled by 1/S.
    w_y = 1 / S * best_boxes[..., 2:4]
    
    # Concatenate the converted x, y, width, and height into one tensor.
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    
    # For class predictions, take the class scores (first 20 numbers) and find the class with highest score.
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    
    # Determine the best confidence score between the two boxes for each cell.
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    
    # Concatenate the predicted class, best confidence, and converted bounding boxes into one tensor.
    # The final format is: [predicted_class, best_confidence, x, y, width, height]
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds




def cellboxes_to_boxes(out, S=7):
    """
    Converts YOLO model outputs (organized per grid cell) to a list of bounding boxes.
    
    Parameters:
        out (tensor): The output tensor from the model.
        S (int): The grid size (default is 7 for a 7x7 grid).
        
    Returns:
        list: A list containing bounding boxes for each example in the batch.
              Each bounding box is a list: [predicted_class, confidence, x, y, width, height].
    """
    # Convert cell-based predictions to image-relative bounding boxes.
    # The output is reshaped to (batch_size, S * S, -1) where -1 infers the remaining dimensions.
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    
    # Ensure the predicted class indices are of type long (integers)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    
    # Initialize a list to store bounding boxes for each example in the batch.
    all_bboxes = []

    # Iterate over each example in the batch.
    for ex_idx in range(out.shape[0]):
        bboxes = []
        # Iterate over each bounding box (each grid cell produces one box)
        for bbox_idx in range(S * S):
            # Convert the tensor values to a Python list of numbers.
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        # Append the list of boxes for this example to the final list.
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves the model's state to a file.
    
    Parameters:
        state (dict): A dictionary containing the model's state_dict and optimizer state_dict.
        filename (str): The file path to save the checkpoint.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)  # Use PyTorch's save functionality to write the state to disk.

def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads a model and optimizer state from a checkpoint.
    
    Parameters:
        checkpoint (dict): A dictionary containing the model and optimizer states.
        model (nn.Module): The model into which to load the state.
        optimizer (Optimizer): The optimizer into which to load the state.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])  # Load model parameters.
    optimizer.load_state_dict(checkpoint["optimizer"])  # Load optimizer state.
