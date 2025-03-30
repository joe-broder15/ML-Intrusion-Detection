def tune_hyperparameters(df, model_class, param_grid, feature_columns=None, label_column='label', 
                   scoring='f1', n_splits=5, random_state=42, n_jobs=-1, verbose=1, return_train_score=True):
    """
    Performs hyperparameter tuning for any sklearn model using grid search with cross-validation.
    
    Parameters:
        df (pandas.DataFrame): The dataset to use for tuning.
        model_class (class): Any sklearn estimator class (not instantiated).
        param_grid (dict): Dictionary with parameters names as keys and lists of parameter values to try.
        feature_columns (list): List of feature column names. If None, uses all columns except label_column.
        label_column (str): Name of the column containing the target labels.
        scoring (str or callable): Strategy to evaluate the performance of the model on the test set.
        n_splits (int): Number of cross-validation splits.
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of jobs to run in parallel (-1 means using all processors).
        verbose (int): Verbosity level (0: no output, 1: progress bar, >1: detailed output).
        return_train_score (bool): If True, also returns scores on training set.
        
    Returns:
        best_params (dict): Dictionary containing the best hyperparameters.
        best_score (float): The best cross-validated score.
        cv_results (dict): Full results from the grid search cross-validation.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score
    import numpy as np
    
    # Handle feature columns
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != label_column]
    
    # Extract features and target
    X = df[feature_columns].values
    y = df[label_column].values
    
    # Define the scoring metric
    if scoring == 'f1':
        scorer = make_scorer(f1_score)
    else:
        scorer = scoring
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize the model
    model = model_class(random_state=random_state) if hasattr(model_class, 'random_state') else model_class()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=return_train_score
    )
    
    print(f"Starting grid search with {np.prod([len(v) for v in param_grid.values()])} parameter combinations...")
    grid_search.fit(X, y)
    
    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\nBest Score ({scoring}): {best_score:.4f}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Return the best parameters, best score, and full results
    return best_params, best_score, grid_search.cv_results_
