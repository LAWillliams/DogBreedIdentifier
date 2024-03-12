from flask import Flask, render_template, request
from model import predict

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_breed():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image', 400

    # Save the uploaded image temporarily
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Perform breed prediction
    predicted_breed = predict.predict_breed(image_path, 'model_checkpoint.pth')

    # Render the prediction result template
    return render_template('result.html', predicted_breed=predicted_breed)


if __name__ == '__main__':
    app.run()