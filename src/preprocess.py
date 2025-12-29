import os
import pandas as pd


RAW_DATA_PATH = "data/raw/insurance.csv"
PROCESSED_DATA_PATH = "data/processed/insurance_processed.csv"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing:
    - handle categorical variables using one-hot encoding
    - no feature engineering
    - no scaling (can be added later if needed)
    """

    # Separate target
    target_column = "charges"
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Combine features and target
    processed_df = pd.concat([X_encoded, y], axis=1)

    return processed_df


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Preprocess
    processed_df = preprocess_data(df)

    # Save processed data
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    print(f"Shape: {processed_df.shape}")


if __name__ == "__main__":
    main()
