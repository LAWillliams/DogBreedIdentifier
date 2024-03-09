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

    # Dictionary with breed annotations
    breed_annotations = {
        "n02085620": "Chihuahua",
        "n02085782": "Japanese Spaniel",
        "n02085936": "Maltese Dog",
        "n02086079": "Pekinese",
        "n02086240": "Shih-Tzu",
        "n02086646": "Blenheim Spaniel",
        "n02086910": "Papillon",
        "n02087046": "Toy Terrier",
        "n02087394": "Rhodesian Ridgeback",
        "n02088094": "Afghan Hound",
        "n02088238": "Basset",
        "n02088364": "Beagle",
        "n02088466": "Bloodhound",
        "n02088632": "Bluetick",
        "n02089078": "Black and Tan Coonhound",
        "n02089867": "Walker Hound",
        "n02089973": "English Foxhound",
        "n02090379": "Redbone",
        "n02090622": "Borzoi",
        "n02090721": "Irish Wolfhound",
        "n02091032": "Italian Greyhound",
        "n02091134": "Whippet",
        "n02091244": "Ibizan Hound",
        "n02091467": "Norwegian Elkhound",
        "n02091635": "Otterhound",
        "n02091831": "Saluki",
        "n02092002": "Scottish Deerhound",
        "n02092339": "Weimaraner",
        "n02093256": "Staffordshire Bullterrier",
        "n02093428": "American Staffordshire Terrier",
        "n02093647": "Bedlington Terrier",
        "n02093754": "Border Terrier",
        "n02093859": "Kerry Blue Terrier",
        "n02093991": "Irish Terrier",
        "n02094114": "Norfolk Terrier",
        "n02094258": "Norwich Terrier",
        "n02094433": "Yorkshire Terrier",
        "n02095314": "Wire Haired Fox Terrier",
        "n02095570": "Lakeland Terrier",
        "n02095889": "Sealyham Terrier",
        "n02096051": "Airedale",
        "n02096177": "Cairn",
        "n02096294": "Australian Terrier",
        "n02096437": "Dandie Dinmont",
        "n02096585": "Boston Bull",
        "n02097047": "Miniature Schnauzer",
        "n02097130": "Giant Schnauzer",
        "n02097209": "Standard Schnauzer",
        "n02097298": "Scotch Terrier",
        "n02097474": "Tibetan Terrier",
        "n02097658": "Silky Terrier",
        "n02098105": "Soft Coated Wheaten Terrier",
        "n02098286": "West Highland White Terrier",
        "n02098413": "Lhasa",
        "n02099267": "Flat Coated Retriever",
        "n02099429": "Curly Coated Retriever",
        "n02099601": "Golden Retriever",
        "n02099712": "Labrador Retriever",
        "n02099849": "Chesapeake Bay Retriever",
        "n02100236": "German Short Haired Pointer",
        "n02100583": "Vizsla",
        "n02100735": "English Setter",
        "n02100877": "Irish Setter",
        "n02101006": "Gordon Setter",
        "n02101388": "Brittany Spaniel",
        "n02101556": "Clumber",
        "n02102040": "English Springer",
        "n02102177": "Welsh Springer Spaniel",
        "n02102318": "Cocker Spaniel",
        "n02102480": "Sussex Spaniel",
        "n02102973": "Irish Water Spaniel",
        "n02104029": "Kuvasz",
        "n02104365": "Schipperke",
        "n02105056": "Groenendael",
        "n02105162": "Malinois",
        "n02105251": "Briard",
        "n02105412": "Kelpie",
        "n02105505": "Komondor",
        "n02105641": "Old English Sheepdog",
        "n02105855": "Shetland Sheepdog",
        "n02106030": "Collie",
        "n02106166": "Border Collie",
        "n02106382": "Bouvier des Flandres",
        "n02106550": "Rottweiler",
        "n02106662": "German Shepherd",
        "n02107142": "Doberman",
        "n02107312": "Miniature Pinscher",
        "n02107574": "Greater Swiss Mountain Dog",
        "n02107683": "Bernese Mountain Dog",
        "n02107908": "Appenzeller",
        "n02108000": "EntleBucher",
        "n02108089": "Boxer",
        "n02108422": "Bull Mastiff",
        "n02108551": "Tibetan Mastiff",
        "n02108915": "French Bulldog",
        "n02109047": "Great Dane",
        "n02109525": "Saint Bernard",
        "n02109961": "Eskimo Dog",
        "n02110063": "Malamute",
        "n02110185": "Siberian Husky",
        "n02110627": "Affenpinscher",
        "n02110806": "Basenji",
        "n02110958": "Pug",
        "n02111129": "Leonberg",
        "n02111277": "Newfoundland",
        "n02111500": "Great Pyrenees",
        "n02111889": "Samoyed",
        "n02112018": "Pomeranian",
        "n02112137": "Chow",
        "n02112350": "Keeshond",
        "n02112706": "Brabancon Griffon",
        "n02113023": "Pembroke",
        "n02113186": "Cardigan",
        "n02113624": "Toy Poodle",
        "n02113712": "Miniature Poodle",
        "n02113799": "Standard Poodle",
        "n02113978": "Mexican Hairless",
        "n02115641": "Dingo",
        "n02115913": "Dhole",
        "n02116738": "African Hunting Dog"
    }

    # Retrieve breed ID using the predicted index
    breed_id = list(breed_annotations.keys())[predicted.item()]

    # Retrieves the breed name using the breed ID
    predicted_breed = breed_annotations[breed_id]

    return predicted_breed
