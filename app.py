from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS, cross_origin
# Load the XGBoost model
model_filename = 'wind_cd_17_aug.pkl'
loaded_model = joblib.load(model_filename)

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["GET", "POST"])
def index():
    return jsonify("input:'get request sent '")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            print("^^^^^^^^^^^")
            # Get the JSON data from the request
            data = request.json
        
            Shot_Distance = data.get('shot_distance')
            Wind_Speed = data.get('wind_speed')
            Angle = data.get('angle')
            
            print(Shot_Distance)

            # Convert the JSON data to a DataFrame
            new_sample = pd.DataFrame([{"shot_distance":Shot_Distance,"wind_speed":Wind_Speed,"new_angle_column":Angle}])
            
    
            
            print(new_sample ,'*********')

            # Make predictions on the new dataset using the loaded model
            predictions = loaded_model.predict((new_sample))
            print(predictions)

            # Return the predictions as JSON
            return jsonify(predictions.tolist())

        except Exception as e:
            return jsonify({'error1111': str(e)})
    return jsonify("input:'get request sent '")


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)    
