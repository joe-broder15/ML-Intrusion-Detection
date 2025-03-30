from sklearn.feature_selection import RFECV, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

def select_features_by_chi2(df, categorical_features, target_column='label', independence_threshold=0.05, 
                           max_features=None, verbose=True, eliminate_dependent=True):
    """
    Identifies and ranks categorical features based on their Chi-squared statistics
    with the target label, returning features sorted by Chi-squared strength.
    Also eliminates features that are not independent from already selected features.
    
    Parameters:
        df (pandas.DataFrame): The dataset containing features and target variable.
        categorical_features (list): List of categorical feature column names to consider.
        target_column (str): Name of the target variable column. Default is 'label'.
        independence_threshold (float): p-value threshold below which features are considered
                                       dependent (not independent). Default is 0.05.
        max_features (int, optional): Maximum number of features to select. If None, selects
                                     as many as pass the independence test.
        verbose (bool): Whether to print progress information. Default is True.
        eliminate_dependent (bool): Whether to eliminate features that are dependent on
                                   already selected features. Default is True.
    
    Returns:
        list: Selected categorical features sorted by Chi-squared strength (highest to lowest)
              with dependent features removed (if eliminate_dependent is True).
    """
    # Ensure target column is not in the categorical features list
    cat_features = [f for f in categorical_features if f != target_column]
    
    # Extract categorical features and target
    X_cat = df[cat_features]
    y = df[target_column]
    
    # Calculate chi-squared scores for each feature against the target
    chi2_scores, p_values = chi2(X_cat, y)
    
    # Pair each feature with its chi-squared score
    feature_scores = list(zip(cat_features, chi2_scores))
    
    # Sort features by chi-squared score (highest to lowest)
    sorted_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    
    # Initialize list of selected features
    selected_features = []
    
    # Loop through features sorted by chi-squared score
    for feature, score in sorted_scores:
        # If we've reached maximum features, stop
        if max_features is not None and len(selected_features) >= max_features:
            break
            
        # Only check for independence if eliminate_dependent is True
        if not eliminate_dependent:
            selected_features.append(feature)
            if verbose:
                print(f"Selected feature '{feature}' (chi2 score: {score:.4f})")
            continue
            
        # Only add feature if it's independent from already selected features
        if not selected_features:
            # If no features selected yet, add the first one
            selected_features.append(feature)
            if verbose:
                print(f"Selected feature '{feature}' (chi2 score: {score:.4f})")
        else:
            # Check independence with already selected features
            should_select = True
            for selected_feature in selected_features:
                # Create contingency table between the two categorical features
                contingency = pd.crosstab(df[feature], df[selected_feature])
                
                # Perform chi-squared test of independence
                _, p_value, _, _ = chi2_contingency(contingency)
                
                # If p-value is less than threshold, features are not independent
                if p_value < independence_threshold:
                    if verbose:
                        print(f"Dropping feature '{feature}' (chi2 score: {score:.4f}) due to dependency (p-value: {p_value:.4f}) with already selected feature '{selected_feature}'")
                    should_select = False
                    break
            
            if should_select:
                selected_features.append(feature)
                if verbose:
                    print(f"Selected feature '{feature}' (chi2 score: {score:.4f})")
    
    return selected_features

def select_features_by_correlation(df, feature_columns, categorical_features=None, target_column='label', correlation_threshold=0.8):
    """
    Identifies and ranks numeric features based on their absolute Pearson correlation
    with the target label, returning features sorted by correlation strength.
    Also eliminates highly correlated features among themselves.
    
    Parameters:
        df (pandas.DataFrame): The dataset containing features and target variable.
        feature_columns (list): List of feature column names to consider.
        categorical_features (list, optional): List of categorical features to exclude.
        target_column (str): Name of the target variable column. Default is 'label'.
        correlation_threshold (float): Threshold above which features are considered
                                     highly correlated. Default is 0.8.
    
    Returns:
        list: Selected features sorted by correlation strength (highest to lowest)
              with highly correlated features removed.
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
    
    # Get correlation matrix between features
    correlation_matrix = df[numeric_features].corr().abs()
    
    # Initialize list of selected features
    selected_features = []
    
    # Loop through features sorted by target correlation
    for feature, corr_value in sorted_correlations:
        # Only add feature if it's not highly correlated with any already selected feature
        if not selected_features:
            # If no features selected yet, add the first one
            selected_features.append(feature)
        else:
            # Check correlation with already selected features
            should_select = True
            for selected_feature in selected_features:
                if correlation_matrix.loc[feature, selected_feature] > correlation_threshold:
                    # Skip this feature as it's highly correlated with an already selected one
                    print(f"Dropping feature '{feature}' (target corr: {corr_value:.4f}) due to high correlation ({correlation_matrix.loc[feature, selected_feature]:.4f}) with already selected feature '{selected_feature}'")
                    should_select = False
                    break
            
            if should_select:
                selected_features.append(feature)
    
    return selected_features



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