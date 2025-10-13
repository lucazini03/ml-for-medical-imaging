def calculate_precision(y_test, grid_predictions):
    #precision calculated manually
    false_positives = ((y_test == 0) & (grid_predictions == 1)).sum()
    true_positives = ((y_test == 1) & (grid_predictions == 1)).sum()
    return true_positives / (true_positives + false_positives)