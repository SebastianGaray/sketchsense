import numpy as np

from sketchsense.evaluation.metrics import classification_metrics
from sketchsense.models.baseline import fit_model


def test_baseline_fit_is_reproducible() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(90, 12)).astype(np.float32)
    labels = np.repeat(np.arange(3, dtype=np.uint8), 30)
    first = fit_model(features, labels, 17)
    second = fit_model(features, labels, 17)
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first.intercept_, second.intercept_, rtol=0.0, atol=0.0)


def test_metrics_include_top_three_accuracy() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    probabilities = np.asarray(
        [[0.7, 0.1, 0.1, 0.1], [0.2, 0.4, 0.3, 0.1], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
        dtype=np.float64,
    )
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == 0.5
    assert metrics["top_3_accuracy"] == 0.75
