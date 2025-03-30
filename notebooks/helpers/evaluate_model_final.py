from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def train_and_evaluate_model(train_df, test_df, selected_features, label_column, model=None, model_params=None, stratify=False, n_splits=5):
    """
    Trains any scikit-learn model on the training set and evaluates its performance on the testing set.
    Optionally uses stratification to maintain class distribution.

    Parameters:
      train_df (pd.DataFrame): The training dataset.
      test_df (pd.DataFrame): The testing dataset.
      selected_features (list): List of features to use for training.
      label_column (str): The name of the target label column.
      model (sklearn estimator, optional): A scikit-learn model instance. If None, LogisticRegression is used.
      model_params (dict, optional): Parameters to initialize the model with if model is None.
                                     For LogisticRegression default: 
                                     {'random_state': 42}
      stratify (bool): Whether to use stratified cross-validation during training (default: False)
      n_splits (int): Number of splits for stratified cross-validation if stratify=True (default: 5)

    Returns:
      results (dict): A dictionary containing evaluation metrics:
                      - accuracy: Accuracy score on the test set.
                      - precision: Precision score.
                      - recall: Recall score.
                      - f1: F1 score.
                      - true_positive_rate: Fraction of positive samples correctly classified.
                      - true_negative_rate: Fraction of negative samples correctly classified.
                      - false_positive_rate: Fraction of negative samples incorrectly classified as positive.
                      - false_negative_rate: Fraction of positive samples incorrectly classified as negative.
      model (sklearn estimator): The trained model.
    """
    # Prepare training features and labels.
    X_train = train_df[selected_features].values
    y_train = train_df[label_column].values

    # Prepare testing features and labels.
    X_test = test_df[selected_features].values
    y_test = test_df[label_column].values

    # Initialize the model
    if model is None:
        # Default to LogisticRegression if no model is provided
        if model_params is None:
            model_params = {'random_state': 42}
        model = LogisticRegression(**model_params)
    
    if stratify:
        # Use stratified cross-validation for training
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_model = None
        best_score = -1
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train on this fold
            model_fold = clone(model)
            model_fold.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation set
            score = model_fold.score(X_fold_val, y_fold_val)
            
            # Keep track of best model
            if score > best_score:
                best_score = score
                best_model = model_fold
        
        # Use the best model from cross-validation
        model = best_model
    else:
        # Train the model normally on the full training set
        model.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = model.predict(X_test)
    
    # Compute the confusion matrix and derive rates
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate evaluation metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'true_positive_rate': true_positive_rate,
        'true_negative_rate': true_negative_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
    }
    
    return results, model
