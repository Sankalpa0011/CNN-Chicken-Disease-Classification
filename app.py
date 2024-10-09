import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import cla
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")  # also can use 'dvc repro' for this
    return "Training done successfully"


# @app.route('/predict', methods=['POST'])
# @cross_origin()
# def predictRoute():
#     image = request.json["image"]
#     decodeImage(image, clApp.filename)
#     result = clApp.classifier.predict()
#     return jsonify(result)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    # Check if request contains JSON
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()

    # Check if "image" is in the request body
    if data is None or "image" not in data:
        return jsonify({"error": "No image provided in the request"}), 400

    try:
        image = data["image"]
        decodeImage(image, clApp.filename)  # Decode the image to save it locally

        # Run the prediction
        result = clApp.classifier.predict()

        return jsonify(result), 200  # Return the result as a JSON response

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    app.run(host='0.0.0.0', port=80) #for Azure