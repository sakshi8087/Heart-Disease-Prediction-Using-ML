from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd
app = Flask(__name__)

# Load the model
model = joblib.load('heart_disease_model.pkl')

# Load individual label encoders for each categorical feature
sex_encoder = joblib.load('Sex_encoder.pkl')
cp_encoder = joblib.load('ChestPainType_encoder.pkl')
ecg_encoder = joblib.load('RestingECG_encoder.pkl')
exang_encoder = joblib.load('ExerciseAngina_encoder.pkl')
stslope_encoder = joblib.load('ST_Slope_encoder.pkl')

# Define top features (ensure this matches the features used during training)
top_features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
label_mapping = {0: 'No Heart Disease', 1: 'Heart Disease'}

@app.route('/')
def home():
    return render_template('index.html', features=top_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_values = []
        for feature in top_features:
            value = request.form.get(feature)
            if value is None or value.strip() == '':
                raise ValueError(f"Missing value for {feature}")

            # Encode categorical features using their respective encoders
            if feature == 'Sex':
                valid_classes = sex_encoder.classes_
                if value not in valid_classes:
                    raise ValueError(f"Invalid value for {feature}: '{value}' is not a recognized category. Expected one of {valid_classes}.")
                value = sex_encoder.transform([value])[0]
            elif feature == 'ChestPainType':
                valid_classes = cp_encoder.classes_
                if value not in valid_classes:
                    raise ValueError(f"Invalid value for {feature}: '{value}' is not a recognized category. Expected one of {valid_classes}.")
                value = cp_encoder.transform([value])[0]
            elif feature == 'RestingECG':
                valid_classes = ecg_encoder.classes_
                if value not in valid_classes:
                    raise ValueError(f"Invalid value for {feature}: '{value}' is not a recognized category. Expected one of {valid_classes}.")
                value = ecg_encoder.transform([value])[0]
            elif feature == 'ExerciseAngina':
                valid_classes = exang_encoder.classes_
                if value not in valid_classes:
                    raise ValueError(f"Invalid value for {feature}: '{value}' is not a recognized category. Expected one of {valid_classes}.")
                value = exang_encoder.transform([value])[0]
            elif feature == 'ST_Slope':
                valid_classes = stslope_encoder.classes_
                if value not in valid_classes:
                    raise ValueError(f"Invalid value for {feature}: '{value}' is not a recognized category. Expected one of {valid_classes}.")
                value = stslope_encoder.transform([value])[0]
            else:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {feature}: '{value}' is not a valid number.")
            feature_values.append(value)

        # Convert feature values to DataFrame with proper column names
        input_data = pd.DataFrame(np.array(feature_values).reshape(1, -1), columns=top_features)

        # Make predictions
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0][1]
        predicted_label = label_mapping.get(prediction, 'Unknown')

        result = {
            'prediction': predicted_label,
            'probability': round(prediction_prob * 100, 2)
        }

        return render_template('result.html', result=result)

    except ValueError as ve:
        error_message = f"Invalid input: {ve}"
        return render_template('error.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True)
