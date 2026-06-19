from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import statistics

app = Flask(__name__)

# Load dataset
data_path = "C:/Users/HARISMITA/Downloads/Training.csv"
data = pd.read_csv(data_path).dropna(axis=1)

# Encode target column
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train models
final_svm_model = SVC(probability=True)
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Symptom dictionary
symptoms = X.columns.values

symptom_index = {
    " ".join([i.capitalize() for i in value.split("_")]): index
    for index, value in enumerate(symptoms)
}

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}


def predictDisease(symptoms_input):

    symptoms = symptoms_input.split(",")

    input_data = [0] * len(data_dict["symptom_index"])

    for symptom in symptoms:
        symptom = symptom.strip()

        index = data_dict["symptom_index"].get(symptom, None)

        if index is not None:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][
        final_rf_model.predict(input_data)[0]
    ]

    nb_prediction = data_dict["predictions_classes"][
        final_nb_model.predict(input_data)[0]
    ]

    svm_prediction = data_dict["predictions_classes"][
        final_svm_model.predict(input_data)[0]
    ]

    final_prediction = statistics.mode(
        [rf_prediction, nb_prediction, svm_prediction]
    )

    # Get probabilities for each model
    rf_proba = final_rf_model.predict_proba(input_data)[0]
    nb_proba = final_nb_model.predict_proba(input_data)[0]
    svm_proba = final_svm_model.predict_proba(input_data)[0]

    # Top 5 predictions
    rf_top5 = sorted(
        zip(data_dict["predictions_classes"], rf_proba),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    nb_top5 = sorted(
        zip(data_dict["predictions_classes"], nb_proba),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    svm_top5 = sorted(
        zip(data_dict["predictions_classes"], svm_proba),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    guidance = {

    "Chronic cholestasis": {
        "precautions": [
            "Avoid alcohol and smoking.",
            "Follow regular liver function check-ups.",
            "Take prescribed medications consistently.",
            "Avoid unnecessary medications that may affect the liver.",
            "Seek medical attention if jaundice or severe itching worsens."
        ],
        "medications": [
            "Take medications only as prescribed by your doctor.",
            "Ursodeoxycholic acid may be prescribed in some cases.",
            "Use anti-itch medications if recommended by a healthcare professional."
        ],
        "diet": [
            "Eat a low-fat diet.",
            "Include fruits and vegetables daily.",
            "Choose whole grains and high-fiber foods.",
            "Drink plenty of water.",
            "Avoid fried foods and processed foods."
        ]
    },

    "Heart attack": {
        "precautions": [
            "Monitor blood pressure regularly.",
            "Avoid smoking and alcohol.",
            "Exercise regularly as advised by your doctor."
        ],
        "medications": [
            "Take medications exactly as prescribed.",
            "Follow up with a cardiologist regularly."
        ],
        "diet": [
            "Reduce saturated fats.",
            "Eat fruits, vegetables, and whole grains.",
            "Limit salt and processed foods."
        ]
    }
    }
    disease_guidance = guidance.get(
    final_prediction,
    {
        "precautions": [
            "Consult a healthcare professional.",
            "Do not rely solely on prediction results."
        ],
        "medications": [
            "Take medicines only if prescribed."
        ],
        "diet": [
            "Maintain a healthy balanced diet."
        ]
    }
    )
    
    return {
       "rf_model_prediction": rf_prediction,
       "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
        "rf_probabilities": rf_top5,
        "nb_probabilities": nb_top5,
         "svm_probabilities": svm_top5,

        "precautions": disease_guidance["precautions"],
        "medications": disease_guidance["medications"],
        "diet_recommendations": disease_guidance["diet"]

    }


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        symptoms_input = request.form["symptoms"]

        predictions = predictDisease(symptoms_input)

        return render_template(
            "index.html",
            predictions=predictions,
            symptoms_input=symptoms_input
        )

    return render_template(
        "index.html",
        predictions=None
    )


if __name__ == "__main__":
    app.run(debug=True)