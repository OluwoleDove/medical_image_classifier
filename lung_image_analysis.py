import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define transformations (resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resizing the images
    transforms.ToTensor(),          # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean/std
])

# Point the directory to where your dataset is stored (replace 'path_to_dataset' with your actual path)
dataset_dir = 'Lung_XRay_Image'

# Load the dataset
dataset = datasets.ImageFolder(dataset_dir, transform=transform)

# Split dataset into training and testing (you can also create a validation set if needed)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
