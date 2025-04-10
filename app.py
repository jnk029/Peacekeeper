from flask import Flask, request, jsonify, render_template
from stress_model import StressDetectionModel

app = Flask(__name__)
model = StressDetectionModel("config.json")
model.load_saved_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    result = model.predict(text)[0]
    return jsonify({
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "class_scores": result["class_scores"]
    })

if __name__ == "__main__":
    app.run(debug=True)

