from typing import List, Tuple  # Import type hints for better code clarity
import pandas as pd
import numpy as np

def preprocess_data(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    categorical_features: List[str], 
    features_to_transform: List[str],
    columns_to_drop: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Preprocesses training and testing datasets by:
    1. Dropping unnecessary columns
    2. Applying log transformation to numeric features
    3. One-hot encoding categorical features
    
    Args:
        train_data: Training dataset
        test_data: Testing dataset
        categorical_features: List of categorical features to one-hot encode
        features_to_transform: List of numeric features to apply log transformation
        columns_to_drop: Optional list of columns to remove
        
    Returns:
        Tuple containing:
        - Processed training dataset
        - Processed testing dataset
        - List of categorical feature columns after one-hot encoding
    """
    # Create copies to avoid modifying the original dataframes
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # Clean both datasets by removing columns that are not needed for modeling
    if columns_to_drop:
        for df in [train_df, test_df]:
            for col in columns_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
    
    # Process numeric features on each dataset independently
    train_df = process_numeric_features(train_df, features_to_transform)
    test_df = process_numeric_features(test_df, features_to_transform)
    
    # Process categorical features on each dataset independently
    train_df, train_categorical_features = process_categorical_features(train_df, categorical_features)
    test_df, test_categorical_features = process_categorical_features(test_df, categorical_features)
    
    # Calculate the union of the new categorical feature sets from training and testing datasets
    common_categorical_features = [i for i in train_categorical_features if i in test_categorical_features]
    
    # Output the shapes of the processed datasets and confirm that all features are numeric
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    print(f"Any non-numeric columns remaining in Training data: {any(not pd.api.types.is_numeric_dtype(train_df[col]) for col in train_df.columns)}")
    
    return train_df, test_df, common_categorical_features

def process_numeric_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Applies a natural logarithm transformation (ln(x+1)) to each specified numeric feature 
    in the provided dataframe. This helps to normalize the distribution of features and 
    mitigate the effect of extreme values.
    """
    
    # apply the log transformation to the features that were determined in EDA
    for feature in features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])
    print("Log transformation applied to numeric features (if present) in the dataset")

    return df

def process_categorical_features(df: pd.DataFrame, cat_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encodes specified categorical features in the provided dataframe.
    Processes each feature independently. If a feature is missing, a warning is printed.
    
    Returns:
        Tuple containing:
        - Updated dataframe with one-hot encoded categorical features
        - List of the names of all the new categorical features
    """
    for feature in cat_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature].astype(str), prefix=feature)
            df = df.drop(columns=[feature])
            df = pd.concat([df, dummies], axis=1)
        else:
            print(f"Warning: '{feature}' not found in the dataframe; skipping one-hot encoding for this feature.")
    print("One-hot encoding applied to categorical features in the dataset")
    updated_dummy_cols = [col for col in df.columns if any(col.startswith(f"{feature}_") for feature in cat_features)]
    print("Updated categorical feature columns:", updated_dummy_cols)
    return df, updated_dummy_cols
