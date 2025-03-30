from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt


def select_features_by_correlation(df, feature_columns, categorical_features=None, target_column='label'):
    """
    Identifies and ranks numeric features based on their absolute Pearson correlation
    with the target label, returning features sorted by correlation strength.
    
    Parameters:
        df (pandas.DataFrame): The dataset containing features and target variable.
        feature_columns (list): List of feature column names to consider.
        categorical_features (list, optional): List of categorical features to exclude.
        target_column (str): Name of the target variable column. Default is 'label'.
    
    Returns:
        list: All features sorted by correlation strength (highest to lowest).
    """
    # Ensure we're working with a clean list of numeric features
    if categorical_features is not None:
        numeric_features = [col for col in feature_columns if col not in categorical_features]
    else:
        numeric_features = feature_columns
    
    # Remove target column if it's in the numeric features list
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    # Calculate correlations between each feature and the target
    correlations = []
    for feature in numeric_features:
        try:
            corr_value = abs(df[feature].corr(df[target_column]))
            correlations.append((feature, corr_value))
        except KeyError:
            print(f"Error: Unable to calculate correlation for feature '{feature}'.")
    
    # Sort features by correlation strength (highest to lowest)
    sorted_correlations = sorted(correlations, key=lambda item: item[1], reverse=True)
    
    # Return list of features sorted by correlation strength
    return [feature for feature, _ in sorted_correlations]



# FUNCTIONS FOR PERFORMING RFE WITH CROSS-VALIDATION

def perform_rfe(df, feature_columns, model_class=None, model_params=None, label_column='label', n_features_to_select=10, 
                step=1, cv=5, scoring='f1', stratified=True, random_state=42, verbose=1):
    """
    Performs Recursive Feature Elimination (RFE) with cross-validation to select the optimal features.
    
    Parameters:
        df (pandas.DataFrame): The dataset containing features and target variable.
        feature_columns (list): List of feature column names to consider for selection.
        model_class (class): Any sklearn estimator class (not instantiated). If None, uses LogisticRegression.
        model_params (dict): Dictionary of parameters to pass to the model constructor. If None, uses default params.
        label_column (str): Name of the target variable column.
        n_features_to_select (int or float): Number of features to select. If float between 0 and 1,
                                            it represents the proportion of features to select.
        step (int or float): Number of features to remove at each iteration. If float between 0 and 1,
                             it represents the proportion of features to remove at each iteration.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric to use for feature selection.
        stratified (bool): Whether to use stratified cross-validation. Default is True.
        random_state (int): Random seed for reproducibility.
        verbose (int): Controls verbosity of output.
        
    Returns:
        selected_features (list): List of selected feature names.
        rfe_cv (RFECV object): The fitted RFECV object for further inspection.
        cv_results (dict): Cross-validation results.
    """
    # Extract features and target
    X = df[feature_columns]
    y = df[label_column]
    
    # Handle default model parameters
    if model_params is None:
        model_params = {}
    
    # Set default model class if none provided
    if model_class is None:
        model_class = LogisticRegression
        # Add some default parameters for LogisticRegression if not specified
        if 'max_iter' not in model_params:
            model_params['max_iter'] = 5000
        if 'random_state' not in model_params:
            model_params['random_state'] = random_state
    
    # Initialize the model with provided parameters
    estimator = model_class(**model_params)
    
    # Set up cross-validation strategy
    if stratified and isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Initialize RFECV
    rfe_cv = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=n_features_to_select,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose
    )
    
    # Fit RFECV
    rfe_cv.fit(X, y)
    
    # Get selected features
    selected_features = [feature for feature, selected in zip(feature_columns, rfe_cv.support_) if selected]
    
    
    # Prepare CV results
    cv_results = {
        'n_features': rfe_cv.n_features_,
        'cv_results_': rfe_cv.cv_results_,  # Using cv_results_ instead of grid_scores_ which is deprecated
        'ranking': rfe_cv.ranking_,
        'support': rfe_cv.support_
    }
    
    return selected_features, rfe_cv, cv_results

def pretty_print_rfecv_results(results):
    """
    Pretty prints the results of the RFECV feature selection process.
    
    Parameters:
        results (tuple): Tuple containing (selected_features, rfe_cv, cv_results)
    """
    selected_features, rfe_cv, cv_results = results
    
    print("\n" + "="*60)
    print(" "*20 + "FEATURE SELECTION RESULTS")
    print("="*60)
    
    # Print optimal number of features
    print(f"Optimal number of features: {cv_results['n_features']}")
    
    # Print selected features
    print("\nSelected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    # Print cross-validation scores
    mean_scores = rfe_cv.cv_results_['mean_test_score']
    std_scores = rfe_cv.cv_results_['std_test_score']
    
    print(f"\nBest cross-validation score: {mean_scores.max():.4f} ± {std_scores[mean_scores.argmax()]:.4f}")
    print(f"Cross-validation scoring metric: {rfe_cv.scoring}")
    
    print("="*60 + "\n")