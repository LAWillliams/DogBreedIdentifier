import torch
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

def preprocess_dataset(dataset_path):
    # Parse the dataset
    data = parse_dataset(dataset_path)

    # Clean the dataset
    cleaned_data = clean_dataset(data)

    # Wrangle the dataset
    wrangled_data = wrangle_dataset(cleaned_data)

    return wrangled_data

# Example usage of dataset preprocessing functions
# dataset_path = 'path/to/your/dataset.csv'
# preprocessed_data = preprocess_dataset(dataset_path)
# # Further processing or utilization of preprocessed_data

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(r'C:\Users\lukea\PycharmProjects\dogBreedIdentifier\data\images', transform=train_transforms)
val_dataset = datasets.ImageFolder(r'C:\Users\lukea\PycharmProjects\dogBreedIdentifier\data\images', transform=val_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

# Load a pre-trained model and modify it for our use case
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def parse_dataset(dataset_path):
    # Implement parsing logic based on the dataset format
    # Example: Parse dataset from a CSV file
    import csv
    data = []
    with open(dataset_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


def clean_dataset(data):
    # Implement cleaning logic based on the dataset
    # Example: Remove rows with missing values
    cleaned_data = []
    for row in data:
        if all(row):
            cleaned_data.append(row)
    return cleaned_data


def wrangle_dataset(data):
    # Implement wrangling logic based on the dataset
    # Example: Convert categorical variables to numerical
    wrangled_data = []
    for row in data:
        # Perform wrangling operations
        wrangled_row = [float(value) if value.isdigit() else value for value in row]
        wrangled_data.append(wrangled_row)
    return wrangled_data

def explore_dataset(dataset):
    # Explore the dataset
    print("Dataset size:", len(dataset))
    print("Number of classes:", len(dataset.classes))
    print("Class labels:", dataset.classes)

    # Visualize sample images from the dataset
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    axs = axs.flatten()
    for i in range(10):
        img, label = dataset[i]
        axs[i].imshow(img.permute(1, 2, 0))
        axs[i].set_title(dataset.classes[label])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def prepare_data(dataset):
    # Prepare the data for training
    data = []
    labels = []
    for img, label in dataset:
        data.append(img.numpy())
        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=25):
    best_val_accuracy = 0.0  # Initialize the best validation accuracy

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_running_corrects.double() / len(val_dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print()

        # Check if this is the best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model_dog_breeds.pth')

    return model, best_val_accuracy

# Function for evaluating accuracy of the trained model
def evaluate_model(model, test_dataset):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")

# Explore the training dataset
explore_dataset(train_dataset)

# Prepare the training data
train_data, train_labels = prepare_data(train_dataset)

# Train the model and get the best validation accuracy
model_ft, best_val_accuracy = train_model(model, criterion, optimizer, num_epochs=25)

# Save the trained model
torch.save({
    'model_state_dict': model_ft.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_accuracy': best_val_accuracy,
    # Include any other relevant information
}, 'model_checkpoint.pth')

