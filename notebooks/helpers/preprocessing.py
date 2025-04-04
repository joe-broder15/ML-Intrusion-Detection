from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def preprocess_data(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    categorical_features: List[str], 
    features_to_transform: List[str],
    columns_to_drop: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Preprocesses training and testing datasets for machine learning.
    
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
    logger = logging.getLogger(__name__)
    
    # Step 1: Create copies and drop specified columns
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # Ensure we're not dropping target column if present
    if columns_to_drop and 'label' in train_df.columns and 'label' in columns_to_drop:
        logger.warning("Removing 'label' from columns_to_drop to prevent dropping target variable")
        columns_to_drop = [col for col in columns_to_drop if col != 'label']
    
    if columns_to_drop:
        train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns])
        test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])
    
    # Step 2: Keep only common columns
    common_columns = list(set(train_df.columns) & set(test_df.columns))
    logger.info(f"Number of common columns: {len(common_columns)}")
    
    train_df = train_df[common_columns]
    test_df = test_df[common_columns]
    
    # Step 3: Filter feature lists to include only common columns
    categorical_features = [col for col in categorical_features if col in common_columns]
    features_to_transform = [col for col in features_to_transform if col in common_columns]
    
    # Step 4: Apply log transformation to numeric features
    for feature in features_to_transform:
        train_df[feature] = np.log1p(train_df[feature])
        test_df[feature] = np.log1p(test_df[feature])
    
    # Step 5: One-hot encode categorical features
    all_encoded_columns = []
    
    for feature in categorical_features:
        # Handle NaN values by converting to string "nan"
        train_df[feature] = train_df[feature].fillna("nan").astype(str)
        test_df[feature] = test_df[feature].fillna("nan").astype(str)
        
        # Create dummy variables (only train categories are kept)
        train_dummies = pd.get_dummies(train_df[feature], prefix=feature)
        all_encoded_columns.extend(train_dummies.columns.tolist())
        
        # For test set, create dummies with same categories as train
        test_dummies = pd.get_dummies(test_df[feature], prefix=feature)
        
        # Ensure test has same dummy columns as train
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
        
        # Drop original column and add dummies
        train_df = train_df.drop(columns=[feature])
        test_df = test_df.drop(columns=[feature])
        
        train_df = pd.concat([train_df, train_dummies], axis=1)
        test_df = pd.concat([test_df, test_dummies], axis=1)
    
    # Step 6: Normalize numeric features
    # Identify numeric columns that aren't one-hot encoded or label
    numeric_cols = [col for col in train_df.columns 
                   if col not in all_encoded_columns 
                   and col != 'label'
                   and pd.api.types.is_numeric_dtype(train_df[col])]
    
    if numeric_cols:
        scaler = StandardScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    
    logger.info(f"Processed shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    
    return train_df, test_df, all_encoded_columns