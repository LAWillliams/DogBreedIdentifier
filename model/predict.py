import torch
from torchvision import transforms, models
from PIL import Image

def preprocess_image(image_path):
    # Define the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load image
    image = Image.open(image_path)
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def load_model(model_path):
    # Assuming you used a ResNet-50 model; adjust as necessary
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    # Update the classifier to match the number of classes you trained on
    model.fc = torch.nn.Linear(num_ftrs, number_of_dog_breeds)  # Update `number_of_dog_breeds` accordingly
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model


def predict_breed(image_path, model_path):
    # Load the model
    model = load_model(model_path)
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    # Make the prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Here you should translate the `predicted` index to the corresponding breed name
    # This requires having a mapping from class indices to breed names, which you should define
    # For example:
    # breed_names = ['breed1', 'breed2', ..., 'breedN']  # Define this list based on your dataset
    # predicted_breed = breed_names[predicted.item()]

    return predicted_breed
