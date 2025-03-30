# Import necessary libraries
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np

def perform_model_cv(df, model_class, model_params=None, feature_columns=None, 
                 label_column='label', n_splits=5, random_state=42):
    """
    Performs k-fold cross-validation for any sklearn model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing features and labels
    model_class : sklearn estimator class
        Any sklearn model class (not instantiated)
    model_params : dict, optional
        Dictionary of parameters to pass to the model constructor
    feature_columns : list, optional
        List of feature column names. If None, uses all columns except label_column
    label_column : str, default='label'
        Name of the target/label column
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics with mean and std across folds
    """
    # Handle default model parameters
    if model_params is None:
        model_params = {}
    
    # Handle feature columns
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != label_column]
    
    # Convert to numpy arrays
    X = df[feature_columns].values
    y = df[label_column].values
    
    # Set up Stratified K-Fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize metric lists
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    tpr_list, fpr_list, tnr_list, fnr_list = [], [], [], []
    
    # Iterate over each fold
    for train_index, test_index in kf.split(X, y):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize model with provided parameters
        model = model_class(**model_params)
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        # Calculate confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        tnr_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        fnr_list.append(fn / (fn + tp) if (fn + tp) > 0 else 0)
    
    # Compile results
    results = {
        'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
        'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
        'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)},
        'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
        'true_positive_rate': {'mean': np.mean(tpr_list), 'std': np.std(tpr_list)},
        'false_positive_rate': {'mean': np.mean(fpr_list), 'std': np.std(fpr_list)},
        'true_negative_rate': {'mean': np.mean(tnr_list), 'std': np.std(tnr_list)},
        'false_negative_rate': {'mean': np.mean(fnr_list), 'std': np.std(fnr_list)}
    }
    return results

# Example for backward compatibility
def perform_logistic_regression_cv(df, feature_columns, label_column='label', n_splits=5, random_state=42,
                                  penalty='l2', C=1.0, solver='sag', max_iter=5000, tol=1e-4, n_jobs=-1, 
                                  verbose=False, class_weight=None):
    """
    Wrapper function for backward compatibility.
    Performs k-fold cross-validation specifically for logistic regression.
    """
    model_params = {
        'penalty': penalty,
        'C': C,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbose': verbose,
        'class_weight': class_weight
    }
    
    return perform_model_cv(
        df=df,
        model_class=LogisticRegression,
        model_params=model_params,
        feature_columns=feature_columns,
        label_column=label_column,
        n_splits=n_splits,
        random_state=random_state
    )

def pretty_print_results(results):
    # Iterate through each metric in the results dictionary.
    for metric, values in results.items():
        # Format and print each metric's name, mean, and standard deviation.
        print(f"{metric.replace('_', ' ').capitalize()}: {values['mean']:.4f} (±{values['std']:.4f})")

