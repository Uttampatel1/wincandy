from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS, cross_origin
# Load the XGBoost model
model_filename = 'xgb_model_fin.pkl'
loaded_model = joblib.load(model_filename)

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            print("^^^^^^^^^^^")
            # Get the JSON data from the request
            Shot_Distance = request.form.get("Shot Distance")
            Wind_Speed = request.form.get("Wind Speed")
            Angle = request.form.get("Angle")
            
            print(Shot_Distance)

            # Convert the JSON data to a DataFrame
            new_sample = pd.DataFrame([{"shot_distance":int(Shot_Distance),"wind_speed":int(Wind_Speed),"angle":int(Angle)}])
            print(new_sample ,'*********')

            # Make predictions on the new dataset using the loaded model
            predictions = loaded_model.predict((new_sample))

            # Return the predictions as JSON
            return jsonify(predictions.tolist())

        except Exception as e:
            return jsonify({'error1111': str(e)})
    return jsonify("input:'get request sent '")


if __name__ == '__main__':
    app.run(debug=True)