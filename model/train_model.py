import torch
import torchvision
from torchvision import datasets, models, transforms
import os

# Define transformations for the training data and validation data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder('path/to/train_data', transform=train_transforms)
val_dataset = datasets.ImageFolder('path/to/val_data', transform=val_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

# Load a pre-trained model and modify it for our use case
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=25):
    best_val_accuracy = 0.0  # Initialize the best validation accuracy

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # Training phase (similar to what you have)

        # Validation phase
        model.eval()  # Set model to evaluate mode
        val_running_corrects = 0
        val_running_loss = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_running_corrects.double() / len(val_dataset)

        print(f'Epoch {epoch+1}/{num_epochs} Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Check if this is the best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Save the model as a checkpoint
            torch.save(model.state_dict(), 'best_model_dog_breeds.pth')

    return model, best_val_accuracy


# Train the model and get the best validation accuracy
model_ft, best_val_accuracy = train_model(model, criterion, optimizer, num_epochs=25)


# Save the trained model
torch.save({
    'model_state_dict': model_ft.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_accuracy': best_val_accuracy,
    # Include any other relevant information
}, 'model_checkpoint.pth')

