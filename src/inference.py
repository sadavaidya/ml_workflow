import joblib
import pandas as pd


MODEL_PATH = "models/model.pkl"


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def predict(model, input_df: pd.DataFrame):
    return model.predict(input_df)


def main():
    # Example input (matches training features)
    sample_input = {
        "age": 40,
        "bmi": 30.0,
        "children": 2,
        "sex_male": 1,
        "smoker_yes": 0,
        "region_northwest": 0,
        "region_southeast": 1,
        "region_southwest": 0,
    }

    input_df = pd.DataFrame([sample_input])

    model = load_model()
    prediction = predict(model, input_df)

    print("Prediction:", prediction[0])


if __name__ == "__main__":
    main()
