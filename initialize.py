import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

current_dir = os.getcwd()
dataset_dir = os.path.join(current_dir, 'Lung_XRay_Image')
print(f"{current_dir} \n {dataset_dir}")
# Define the path to your dataset directory
#dataset_dir = './Lung_XRay_Image'

# Check if the dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"Directory not found: {dataset_dir}")
else:
    print(f"Directory exists: {dataset_dir}")

# Define transforms for training and testing
transform_train = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset from the directory using ImageFolder
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform_train)

# Split the dataset into training (80%), validation (10%), and testing (10%) sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Apply the test transforms to validation and test datasets
val_dataset.dataset.transform = transform_test
test_dataset.dataset.transform = transform_test

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print class names to verify the loading
print(f"Class names: {dataset.classes}")