import torch
import torch.nn as nn # Importing PyTorch's nn Module, to build convolutional layers, submodule of torch

# YOLOv1 uses 24 convolutional layers
# The backbone network is called "Darknet" (DarkNet name give by: Joseph Redmon )
# The fully connected layers are defined separately  

architecture_config = [
    
    (7, 64, 2, 3),  # First Convolutional Layer: (Kernel size: 7, Filters: 64, Stride: 2, Padding: 3)
    "M",            # M = Max-Pooling Layer (Reduces spatial dimensions)
    (3, 192, 1, 1), # Second Convolutional Layer
    "M",  
    
    # Series of Convolutional Layers
    (1, 128, 1, 0),  # 1x1 conv (Reduces dimensions)
    (3, 256, 1, 1),  # 3x3 conv (Extracts features)
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    
    # Block with 4 repeated layers
    # Each repeat consists of: 1x1 conv (256 filters), 3x3 conv (512 filters)
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  
    
    # More convolutional layers
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    
    # Block with 2 repeated layers: 1x1 conv (512 filters), 3x3 conv (1024 filters)
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  
    
    # Final Convolutional Layers
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),  # Stride=2 for downsampling
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# Define a Convolutional Block that will be used multiple times in the YOLO model
class CNNBlock(nn.Module):
    """
    A Convolutional Block that will be used multiple times in the YOLOv1 model
    in_channels: RGB: 3 channles
    Leaky ReLU activation function is used
    Batch Normalization: Not included in the original implementation of YOLOv1
    Convolutional layer without bias if we use BatchNorm because BatchNorm handles bias
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        # Batch Normalization layer (helps in faster training and stability)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)          # Negative slope of 0.1

    # Forward pass: Apply Conv → BatchNorm → LeakyReLU
    def forward(self, x):
        return self.leakyrelu((self.batchnorm(self.conv(x))))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        
        super(Yolov1, self).__init__()          # Initialize the parent class (nn.Module)
        self.architecture = architecture_config # Store the architecture configuration (defined earlier)
        self.in_channels = in_channels          # Set the initial number of input channels (3 for an RGB image)                                                       
        self.darknet = self._create_conv_layers(self.architecture) # Create the convolutional (Darknet) layers based on the architecture configuration
        self.fcs = self._create_fcs(**kwargs)                      # Create the fully connected layers (FCs) using additional keyword arguments

    def forward(self, x):
        
        # Pass the input through the convolutional layers (Darknet)
        x = self.darknet(x)
        
        # Flatten the output starting from dimension 1 (preserving the batch dimension)
        # and then pass it through the fully connected layers
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []                    # Initialize an empty list to hold the layers
        in_channels = self.in_channels # Start with the initial number of input channels (3 for an RGB image)

        # Loop through each element in the architecture configuration
        for x in architecture:
            # If the element is a tuple, it represents a convolutional layer
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,               # Number of input channels
                        out_channels=x[1],         # Number of output filters
                        kernel_size=x[0],          # Kernel size of the convolution
                        stride=x[2],               # Stride of the convolution
                        padding=x[3],              # Padding applied to the input
                    )
                ]
                in_channels = x[1] # Update in_channels for the next layer to be the current output channels

            # If the element is a string, it represents a max-pooling layer
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # Add a max-pooling layer with kernel size 2 and stride 2
            
            # If the element is a list, it represents a block of layers to be repeated
            elif type(x) == list:
                conv1 = x[0]        # The first tuple defines the first convolution in the repeat block
                conv2 = x[1]        # The second tuple defines the second convolution in the repeat block
                num_repeats = x[2]  # The integer indicates how many times to repeat this pair of layers

                for _ in range(num_repeats):
                    # First, add the first convolutional block of the repeat
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
    
                    # Then, add the second convolutional block of the repeat,
                    # using the output channels from conv1 as the input channels
                    layers += [
                        CNNBlock(
                            conv1[1],      # Input is the output of the previous CNNBlock (conv1)
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    # Update in_channels for the next layer or repeat block to be conv2's output channels
                    in_channels = conv2[1]

        # Return a sequential container with all the layers added
        return nn.Sequential(*layers)

    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        # Unpack the parameters:
        # S: Grid size (e.g., 7 for a 7x7 grid)
        # B: Number of bounding boxes per grid cell (e.g., 2)
        # C: Number of classes (e.g., 20 for Pascal VOC)
        S, B, C = split_size, num_boxes, num_classes

        # Build a sequential container with the following layers:
        return nn.Sequential(
            # Flatten the output of the convolutional layers into a single vector per image.
            # The expected size here is 1024 * S * S.
            # 1024 is the number of feature maps from the final convolution layer,
            # and S x S is the grid dimension.
            nn.Flatten(),

            # The first fully connected layer maps the flattened vector to 4096 neurons.
            # Note: Number of neurons can be reduced to decrease the number of parameters and computation. Only for experimentation.
            nn.Linear(1024 * S * S, 496),  
            
            # Dropout layer (with dropout probability, to drop neurons)
            # It is included to potentially reduce overfitting
            nn.Dropout(0),
            
            # LeakyReLU activation with a negative slope of 0.1,
            # which allows a small gradient when the input is negative.
            nn.LeakyReLU(0.1),

            # The final fully connected layer maps from 4096 neurons to the final output size.
            # Each grid cell in an S x S grid predicts (C + B*5) values: C = Class probabilities (one for each class),
            # B * 5: For each bounding box, 5 values (x, y, width, height, confidence)
            nn.Linear(496, S * S * (C + B * 5)),

            # To map any real-valued number into a range between 0 and 1
            #nn.Sigmoid() # Not included in the original YOLOv1, just for experimental purposes.
            
        )

# test case
def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)
#test()
