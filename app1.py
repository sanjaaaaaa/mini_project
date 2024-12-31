from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the TensorFlow model using tf.keras (without directly importing keras)
model = tf.keras.models.load_model("deceptive_id_model.keras")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from request
    data = request.json
    username = data.get('Username')
    followers = data.get('Followers')
    posts = data.get('Posts')
    sentiment = data.get('Sentiment')  # Example: Sentiment analysis result
    reported = data.get('Reported')  # Example: "Yes" or "No"
    
    # Preprocess the data (this depends on how the model was trained)
    input_data = np.array([len(username), followers, posts, len(bio_keywords), sentiment == "Positive", reported == "Yes"])
    
    # Predict using the model
    # Make sure the input data is in the shape the model expects
    input_data = np.expand_dims(input_data, axis=0)  # Adding batch dimension
    prediction = model(input_data)  # Use model's callable function for prediction
    is_deceptive = 'Deceptive' if prediction[0] > 0.5 else 'Legit'

    return jsonify({
        'Username': username,
        'Prediction': is_deceptive
    })

if __name__ == '__main__':
    app.run(debug=True,port=8080)
