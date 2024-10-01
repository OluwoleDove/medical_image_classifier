import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import initialize
from model import train_model, device, model

# Initialize the model, loss function, and optimizer
initialize
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model for 10 epochs
train_model(model, initialize.train_loader, initialize.val_loader, criterion, optimizer, num_epochs=10)



def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Test the model
test_model(model, initialize.test_loader)



def imshow(img):
    img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random test images
dataiter = iter(initialize.test_loader)
images, labels = dataiter.next()

# Print images
imshow(torchvision.utils.make_grid(images))

# Print predicted labels
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

print('Predicted:', ' '.join(f'{initialize.dataset.classes[predicted[j]]}' for j in range(len(predicted))))
