from sklearn.feature_selection import RFECV, chi2, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
import random
import math
from tqdm.notebook import tqdm
from helpers.cross_validation import perform_model_cv

def select_features_by_mutual_info_categorical(df, categorical_features, target_column='label', 
                                redundancy_threshold=0.7, verbose=True):
    """
    Selects categorical features based on mutual information with target while removing redundant features.
    
    Parameters:
        df (pandas.DataFrame): Dataset containing features and target variable.
        categorical_features (list): List of categorical feature names to consider.
        target_column (str): Name of the target variable column. Default is 'label'.
        redundancy_threshold (float): Threshold for feature redundancy. Default is 0.7.
        verbose (bool): Whether to print progress information. Default is True.
    
    Returns:
        list: Selected categorical features sorted by mutual information strength.
    """
    # Filter out target column
    cat_features = [feature for feature in categorical_features if feature != target_column]
    
    # Calculate mutual information with target
    X_cat = df[cat_features]
    y = df[target_column]
    mi_scores = mutual_info_classif(X_cat, y)
    
    # Sort features by mutual information
    sorted_features = sorted(zip(cat_features, mi_scores), key=lambda x: x[1], reverse=True)
    
    # Select features
    selected = []
    remaining = [f for f, _ in sorted_features]
    
    while remaining:
        # Select feature with highest MI
        best_feature = remaining[0]
        best_mi = next(mi for f, mi in sorted_features if f == best_feature)
        
        # Add to selected features
        selected.append(best_feature)
        if verbose:
            print(f"Selected feature '{best_feature}' (mutual info score: {best_mi:.4f})")
        
        remaining.remove(best_feature)
        
        # Filter out redundant features
        non_redundant = []
        for feature in remaining:
            try:
                # Calculate MI between this feature and best feature
                X1 = df[best_feature].values.reshape(-1, 1)
                X2 = df[feature].values.reshape(-1, 1)
                mi_between = mutual_info_classif(X1, X2.ravel(), discrete_features=True)[0]
                
                # Get feature's MI with target
                feature_mi = next(mi for f, mi in sorted_features if f == feature)
                
                # Calculate redundancy score
                redundancy_score = mi_between / max(feature_mi, 0.001)
                
                if redundancy_score <= redundancy_threshold:
                    non_redundant.append(feature)
                elif verbose:
                    print(f"Excluding '{feature}' (target MI: {feature_mi:.4f}) due to "
                          f"redundancy (score: {redundancy_score:.4f}) with '{best_feature}'")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compute mutual information for '{feature}' "
                          f"and '{best_feature}': {e}")
                non_redundant.append(feature)
        
        # Update remaining features
        remaining = non_redundant
    
    return selected

def select_features_by_mutual_info_numeric(df, feature_columns, categorical_features=None, 
                                          target_column='label', redundancy_threshold=0.7):
    """
    Selects numeric features based on mutual information with target while removing redundant features.
    
    Parameters:
        df (pandas.DataFrame): Dataset containing features and target variable.
        feature_columns (list): List of feature column names to consider.
        categorical_features (list, optional): List of categorical features to exclude.
        target_column (str): Name of the target variable column. Default is 'label'.
        redundancy_threshold (float): Threshold for feature redundancy. Default is 0.7.
    
    Returns:
        list: Selected features sorted by mutual information strength with target.
    """
    # Filter numeric features
    numeric_features = [col for col in feature_columns 
                        if col != target_column and col not in (categorical_features or [])]
    
    try:
        # Calculate mutual information with target
        X = df[numeric_features]
        y = df[target_column]
        mi_scores = mutual_info_regression(X, y)
        
        # Sort features by mutual information
        feature_mi = list(zip(numeric_features, mi_scores))
        sorted_features = sorted(feature_mi, key=lambda x: x[1], reverse=True)
        
        # Select features
        selected = []
        remaining = [f for f, _ in sorted_features]
        
        while remaining:
            # Select feature with highest MI
            best_feature = remaining[0]
            best_mi = next(mi for f, mi in sorted_features if f == best_feature)
            
            # Add to selected features
            selected.append(best_feature)
            remaining.remove(best_feature)
            
            # Filter out redundant features
            non_redundant = []
            for feature in remaining:
                # Calculate MI between this feature and best feature
                X1 = df[best_feature].values.reshape(-1, 1)
                X2 = df[feature].values.reshape(-1, 1)
                mi_between = mutual_info_regression(X1, X2.ravel(), discrete_features=False)[0]
                
                # Get feature's MI with target
                feature_mi_val = next(mi for f, mi in sorted_features if f == feature)
                
                # Calculate redundancy score
                redundancy_score = mi_between / max(feature_mi_val, 0.001)
                
                if redundancy_score <= redundancy_threshold:
                    non_redundant.append(feature)
                else:
                    print(f"Excluding '{feature}' (target MI: {feature_mi_val:.4f}) due to "
                          f"redundancy (score: {redundancy_score:.4f}) with '{best_feature}'")
            
            # Update remaining features
            remaining = non_redundant
        
        return selected
        
    except KeyError as e:
        raise KeyError(f"Column not found in dataframe: {e}")
    except ValueError as e:
        raise ValueError(f"Error in mutual information calculation: {e}")

def select_features_by_chi2(df, categorical_features, target_column='label', independence_threshold=0.05, 
                           max_features=None, verbose=True, eliminate_dependent=True):
    """
    Selects categorical features based on Chi-squared statistics with the target variable.
    
    Parameters:
        df (pandas.DataFrame): Dataset containing features and target variable.
        categorical_features (list): List of categorical feature names to consider.
        target_column (str): Name of the target variable column. Default is 'label'.
        independence_threshold (float): p-value threshold for feature independence test.
        max_features (int, optional): Maximum number of features to select. Default is None.
        verbose (bool): Whether to print progress information. Default is True.
        eliminate_dependent (bool): Whether to eliminate dependent features. Default is True.
    
    Returns:
        list: Selected categorical features sorted by Chi-squared strength.
    """
    # Filter out target column from features
    cat_features = [feature for feature in categorical_features if feature != target_column]
    
    try:
        # Calculate chi-squared statistics
        X_cat = df[cat_features]
        y = df[target_column]
        chi2_scores, _ = chi2(X_cat, y)
        
        # Sort features by chi-squared score
        sorted_features = sorted(zip(cat_features, chi2_scores), key=lambda x: x[1], reverse=True)
        
        selected = []
        for feature, score in tqdm(sorted_features, desc="Chi2 Feature Selection"):
            # Check if we've reached maximum features
            if max_features and len(selected) >= max_features:
                if verbose:
                    print(f"Reached maximum number of features ({max_features})")
                break
                
            # Skip dependency checks if not eliminating dependent features
            if not eliminate_dependent or not selected:
                selected.append(feature)
                if verbose:
                    print(f"Selected feature '{feature}' (chi2 score: {score:.4f})")
                continue
                
            # Check independence with already selected features
            is_independent = True
            for selected_feature in selected:
                contingency = pd.crosstab(df[feature], df[selected_feature])
                try:
                    _, p_value, _, _ = chi2_contingency(contingency)
                    if p_value < independence_threshold:
                        if verbose:
                            print(f"Excluding '{feature}' (chi2 score: {score:.4f}) "
                                  f"due to dependency (p-value: {p_value:.4f}) with '{selected_feature}'")
                        is_independent = False
                        break
                except ValueError as e:
                    if verbose:
                        print(f"Warning: Could not compute chi2_contingency for '{feature}' and '{selected_feature}': {e}")
                    is_independent = False
                    break
                    
            if is_independent:
                selected.append(feature)
                if verbose:
                    print(f"Selected feature '{feature}' (chi2 score: {score:.4f})")
                    
        return selected
        
    except KeyError as e:
        raise KeyError(f"Column not found in dataframe: {e}")
    except ValueError as e:
        raise ValueError(f"Error in chi2 calculation: {e}. Ensure categorical features are encoded as numbers.")

def select_features_by_correlation(df, feature_columns, categorical_features=None, 
                                  target_column='label', correlation_threshold=0.8):
    """
    Selects numeric features based on correlation with target while removing highly correlated features.
    
    Parameters:
        df (pandas.DataFrame): Dataset containing features and target variable.
        feature_columns (list): List of feature column names to consider.
        categorical_features (list, optional): List of categorical features to exclude.
        target_column (str): Name of the target variable column. Default is 'label'.
        correlation_threshold (float): Threshold for feature correlation. Default is 0.8.
    
    Returns:
        list: Selected features sorted by correlation strength with target.
    """
    # Filter numeric features
    numeric_features = [col for col in feature_columns 
                       if col != target_column and col not in (categorical_features or [])]
    
    # Calculate correlations with target
    correlations = []
    for feature in numeric_features:
        try:
            corr = abs(df[feature].corr(df[target_column]))
            correlations.append((feature, corr))
        except KeyError:
            continue
    
    # Sort by correlation strength
    sorted_features = sorted(correlations, key=lambda x: x[1], reverse=True)
    
    # Calculate correlation matrix
    valid_features = [f for f, _ in correlations]
    corr_matrix = df[valid_features].corr().abs()
    
    # Select features
    selected = []
    for feature, corr_with_target in tqdm(sorted_features, desc="Selecting features", total=len(sorted_features), unit="feature"):
        should_include = True
        for selected_feature in selected:
            corr_between_features = corr_matrix.loc[feature, selected_feature]
            if corr_between_features > correlation_threshold:
                print(f"Excluding '{feature}' (target corr: {corr_with_target:.4f}) "
                      f"due to {corr_between_features:.4f} correlation with '{selected_feature}'")
                should_include = False
                break
        
        if should_include:
            selected.append(feature)
    
    return selected

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

def select_features_by_hill_climbing(df, feature_columns, model_class=None, model_params=None, label_column='label',
                                    min_features=5, max_features=20, max_iterations=100, n_folds=5,
                                    epsilon=0.1, random_state=42, scoring='f1', verbose=1):
    """
    Performs feature selection using hill climbing with epsilon-greedy exploration.
    
    Parameters:
        df (pandas.DataFrame): Dataset with features and target.
        feature_columns (list): Feature column names to consider.
        model_class (class): Any sklearn estimator. Uses LogisticRegression if None.
        model_params (dict): Parameters for model constructor.
        label_column (str): Target variable column name.
        min_features (int): Minimum number of features.
        max_features (int): Maximum number of features.
        max_iterations (int): Maximum iterations for hill climbing.
        n_folds (int): Number of cross-validation folds.
        epsilon (float): Probability of accepting random changes (0-1).
        random_state (int): Random seed.
        scoring (str): Scoring metric ('accuracy', 'precision', 'recall', or 'f1').
        verbose (int): Output verbosity level.
        
    Returns:
        selected_features (list): List of selected feature names.
        results (dict): Results of the feature selection process.
    """
    # Set random seed
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Setup model
    if model_class is None:
        model_class = LogisticRegression
        model_params = model_params or {'max_iter': 5000, 'random_state': random_state}
    else:
        model_params = model_params or {}
        if hasattr(model_class(), 'random_state'):
            model_params['random_state'] = model_params.get('random_state', random_state)
    
    # Validate scoring
    valid_scoring = ['accuracy', 'precision', 'recall', 'f1']
    if scoring not in valid_scoring:
        raise ValueError(f"Scoring must be one of {valid_scoring}")
    
    # CV evaluation function
    def evaluate_feature_set(features):
        if not features:
            return {'accuracy': 0, 'f1_score': 0, 'precision': 0, 'recall': 0}
        
        cv_results = perform_model_cv(
            df=df,
            model_class=model_class,
            model_params=model_params,
            feature_columns=features,
            label_column=label_column,
            n_splits=n_folds,
            random_state=random_state
        )
        
        return {
            'accuracy': cv_results['accuracy']['mean'],
            'f1_score': cv_results['f1']['mean'],
            'precision': cv_results['precision']['mean'],
            'recall': cv_results['recall']['mean']
        }
    
    # Filter valid features
    valid_features = [f for f in feature_columns if f in df.columns]
    if len(valid_features) < len(feature_columns) and verbose > 0:
        print(f"Warning: {len(set(feature_columns) - set(valid_features))} features not found in dataframe")
    feature_columns = valid_features
    
    if not feature_columns:
        raise ValueError("No valid features to select from")
    
    # Adjust min/max features if needed
    min_features = min(min_features, len(feature_columns))
    max_features = min(max_features, len(feature_columns))
    
    # Initialize with random subset
    initial_size = min(int((min_features + max_features) / 2), len(feature_columns))
    current_features = random.sample(feature_columns, initial_size)
    score_key = scoring if scoring == 'accuracy' else f"{scoring}_score"
    
    # Evaluate initial features
    current_result = evaluate_feature_set(current_features)
    current_score = current_result[score_key]
    
    if verbose > 0:
        print(f"Initial feature set ({len(current_features)} features) - {scoring.capitalize()}: {current_score:.4f}")
    
    # Track best solution
    best_features = current_features.copy()
    best_score = current_score
    best_result = current_result.copy()
    
    # Progress tracking
    iterations_range = tqdm(range(max_iterations), desc="Hill Climbing") if verbose > 0 else range(max_iterations)
    
    # Hill climbing iterations
    for iteration in iterations_range:
        # Generate candidate
        if len(current_features) <= min_features:
            mod_type = random.choice([0, 2])  # Add or swap
        elif len(current_features) >= max_features:
            mod_type = random.choice([1, 2])  # Remove or swap
        else:
            mod_type = random.randint(0, 2)  # Any modification
        
        candidate_features = current_features.copy()
        modification = ""
        
        # Apply mutation
        if mod_type == 0:  # Add feature
            available = [f for f in feature_columns if f not in candidate_features]
            if not available:
                continue
            to_add = random.choice(available)
            candidate_features.append(to_add)
            modification = f"Added: {to_add}"
            
        elif mod_type == 1:  # Remove feature
            if len(candidate_features) <= min_features:
                continue
            to_remove = random.choice(candidate_features)
            candidate_features.remove(to_remove)
            modification = f"Removed: {to_remove}"
            
        else:  # Swap feature
            available = [f for f in feature_columns if f not in candidate_features]
            if not available or not candidate_features:
                continue
            to_remove = random.choice(candidate_features)
            to_add = random.choice(available)
            candidate_features.remove(to_remove)
            candidate_features.append(to_add)
            modification = f"Swapped: {to_remove} → {to_add}"
        
        # Evaluate candidate
        candidate_result = evaluate_feature_set(candidate_features)
        candidate_score = candidate_result[score_key]
        
        # Epsilon-greedy acceptance
        accept = (candidate_score > current_score or random.random() < epsilon)
        
        if accept:
            current_features = candidate_features
            current_score = candidate_score
            current_result = candidate_result
            
            if verbose > 0:
                diff = current_score - best_score
                status = "improved" if diff > 0 else "explored"
                print(f"Iter {iteration}: {modification} - {scoring}: {current_score:.4f} ({status})")
            
            if current_score > best_score:
                best_features = current_features.copy()
                best_score = current_score
                best_result = current_result.copy()
    
    # Final results
    if verbose > 0:
        print(f"\nBest feature set ({len(best_features)} features) - {scoring}: {best_score:.4f}")
        if verbose > 1:
            print(f"Features: {', '.join(best_features)}")
    
    # Return results
    results = {
        'n_features': len(best_features),
        'f1_score': best_result['f1_score'],
        'accuracy': best_result['accuracy'],
        'precision': best_result['precision'],
        'recall': best_result['recall'],
        'feature_importances': dict(zip(best_features, [1.0] * len(best_features)))
    }
    
    return best_features, results