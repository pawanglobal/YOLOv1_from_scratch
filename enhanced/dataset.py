import torch  
import os               # For working with file paths (joining directories and file names)
import pandas as pd     # For data manipulation, reading CSV files, etc.
from PIL import Image   # For loading and manipulating images (via the Pillow library)

class VOCDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for the Pascal VOC dataset, inheriting from torch.utils.data.Dataset.
    """
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        Initializes the dataset.
        
        Parameters:
            csv_file (str): Path to a CSV file that contains image file names and corresponding label file names.
            img_dir (str): Directory where the images are stored.
            label_dir (str): Directory where the annotation (label) files are stored.
            S (int): Grid size to split the image (default 7 for a 7x7 grid).
            B (int): Number of bounding boxes per grid cell (default 2).
            C (int): Number of classes (default 20 for Pascal VOC).
            transform (callable, optional): A function/transform to apply to both the image and bounding boxes.
        """
        self.annotations = pd.read_csv(csv_file)  # Read the CSV file into a DataFrame
        self.img_dir = img_dir                    # Store the image directory
        self.label_dir = label_dir                # Store the label directory
        self.transform = transform                # Transformation pipeline (e.g., resizing, tensor conversion)
        self.S = S                                # Grid size (SxS)
        self.B = B                                # Number of bounding boxes per grid cell
        self.C = C                                # Number of classes

    # Return the number of samples in the dataset based on the number of rows in the annotations DataFrame. To create a batch size.
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        # Get the path to the label file for the given index
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []  # Initialize a list to store bounding box information

        # Open the label file and read each line
        with open(label_path) as f:
            for label in f.readlines():
                
                # For each line, split the values and convert each value to float (or int if it represents an integer)
                # Each label line is expected to have: class_label, x, y, width, height.
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                
                # Append the bounding box information to the list as [class_label, x, y, width, height]
                boxes.append([class_label, x, y, width, height])
        
        # Get the image file path using the first column in the CSV for this index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)  # Open the image using Pillow
        boxes = torch.tensor(boxes)   # Convert the list of bounding boxes into a PyTorch tensor for easier manipulation

        # Apply the transformation pipeline to the image, and bounding boxes, if transformation is designed for both)
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Create an empty label matrix with shape (S, S, C + 5*B)
        # Each grid cell will eventually hold:
        # - The objectness score (1 value at index 20)
        # - The bounding box coordinates (4 values from indices 21 to 24)
        # - One-hot encoded class probabilities (first C indices)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # Populate the label matrix based on the bounding boxes
        for box in boxes:
            
            # Convert the bounding box tensor to a list of numbers
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)  # Ensure the class label is an integer
            
            # Determine which grid cell the box's center falls into
            i, j = int(self.S * y), int(self.S * x)
            # Calculate the offset of the box's center within that grid cell
            x_cell, y_cell = self.S * x - j, self.S * y - i
            # Scale the width and height to the grid cell scale
            width_cell, height_cell = width * self.S, height * self.S

            # If no object has been assigned to this grid cell yet (objectness score at index 20 is 0)
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # Set the objectness score to 1
                
                # Create a tensor for the bounding box coordinates (relative to the cell)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates  # Assign these coordinates to the matrix
                label_matrix[i, j, class_label] = 1          # One-hot encode the class label in the matrix

        # Return the transformed image and the label matrix which contains the bounding box and class info
        return image, label_matrix
