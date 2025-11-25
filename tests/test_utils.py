import numpy as np
from src.utils import calculate_metrics


def test_calculate_metrics():
    y_true = np.array([100, 200, 300, 400])
    y_pred = np.array([110, 190, 310, 390])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert 0 <= metrics["R2"] <= 1


def test_calculate_metrics_perfect_predictions():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1.0
