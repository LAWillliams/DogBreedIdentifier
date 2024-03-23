from flask import Flask, render_template, request
from model import predict
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    logger.info('Home page accessed')
    return render_template('index.html')

def get_dashboard_data():
    # Placeholder function to generate or fetch data
    # This should return data in a format that can be easily consumed by your frontend, e.g., JSON
    data = {
        'lineChartData': {
            # Example data format for a line chart
            'labels': ['January', 'February', 'March'],
            'datasets': [{
                'label': 'Dataset 1',
                'data': [10, 20, 30]
            }]
        },
        'barChartData': {
            # Example data format for a bar chart
            'labels': ['Category 1', 'Category 2', 'Category 3'],
            'datasets': [{
                'label': 'Dataset 1',
                'data': [5, 10, 15]
            }]
        },
        'pieChartData': {
            # Example data format for a pie chart
            'labels': ['Segment 1', 'Segment 2', 'Segment 3'],
            'data': [300, 50, 100]
        }
    }
    return data

# Call the function and print the returned data
dashboard_data = get_dashboard_data()
print(dashboard_data)

@app.route('/dashboard')
def dashboard():
    logger.info('Dashboard page accessed')
    # Fetch data for visualizations
    data = get_dashboard_data()
    logger.info(data)
    return render_template('dashboard.html', data=data)

@app.route('/predict', methods=['POST'])
def predict_breed():
    logger.info('Prediction request received')
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image', 400

    # Save the uploaded image temporarily
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Perform breed prediction
    predicted_breed = predict.predict_breed(image_path, 'model/best_model_dog_breeds.pth')

    # Get breed distribution
    breed_counts, breed_percentages = predict.get_breed_distribution('model/best_model_dog_breeds.pth')

    logger.info(f'Predicted breed: {predicted_breed}')
    # Render the prediction result template with breed distribution information
    return render_template('result.html', predicted_breed=predicted_breed,
                           breed_counts=breed_counts, breed_percentages=breed_percentages)

# Runs flask app with security with HTTPS implementation
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, ssl_context='adhoc')