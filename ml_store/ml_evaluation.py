"""Model evaluation and interpretation functions for CatBoost models.

This module provides functions for:
- Feature importance analysis
- SHAP (SHapley Additive exPlanations) values and visualizations
- Partial dependence plots
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from plotly.subplots import make_subplots


def _get_feature_names_from_config(
    config: Dict[str, Any], df: Optional[pl.DataFrame] = None
) -> List[str]:
    """Extract feature names from config.

    Args:
        config: Configuration dictionary
        df: Optional DataFrame to infer features from

    Returns:
        List of feature names
    """
    if "feature_columns" in config and config["feature_columns"]:
        return config["feature_columns"]
    elif df is not None:
        # Infer from DataFrame
        exclude_cols = [config.get("target_column")]
        if "exposure_column" in config:
            exclude_cols.append(config["exposure_column"])
        return [col for col in df.columns if col not in exclude_cols]
    else:
        raise ValueError(
            "Cannot determine feature names from config. Provide feature_columns in config or pass df parameter."
        )


def _get_categorical_indices_from_config(
    config: Dict[str, Any], feature_names: List[str]
) -> List[int]:
    """Extract categorical feature indices from config.

    Args:
        config: Configuration dictionary
        feature_names: List of feature names

    Returns:
        List of categorical feature indices
    """
    categorical_features = config.get("categorical_features", [])
    return [i for i, col in enumerate(feature_names) if col in categorical_features]


def _get_backend_from_config(backend: Optional[str], config: Optional[Dict[str, Any]]) -> str:
    """Helper function to determine plotting backend.

    Priority: explicit backend parameter > config > default (matplotlib)

    Args:
        backend: Explicit backend parameter (if provided)
        config: Config dictionary that may contain "plotting_backend"

    Returns:
        Backend string ("matplotlib" or "plotly")
    """
    if backend is not None:
        return backend
    if config is not None and "plotting_backend" in config:
        return config["plotting_backend"]
    return "matplotlib"


def create_pool_from_config(
    X: np.ndarray,
    y: Optional[np.ndarray],
    config: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    df: Optional[pl.DataFrame] = None,
) -> Pool:
    """Create a CatBoost Pool from data and config.

    This is a convenience function that creates a Pool with proper categorical
    feature handling based on the config file.

    Args:
        X: Feature matrix
        y: Target array (optional, can be None for prediction)
        config: Configuration dictionary
        feature_names: List of feature names (optional, will be extracted from config)
        df: Optional DataFrame to infer feature names from

    Returns:
        CatBoost Pool object

    Example:
        # Create pool for evaluation
        pool = create_pool_from_config(X_test, y_test, config)
        importance = get_feature_importance(model, config=config, pool=pool)
    """
    if feature_names is None:
        feature_names = _get_feature_names_from_config(config, df)

    cat_indices = _get_categorical_indices_from_config(config, feature_names)

    return Pool(X, y, cat_features=cat_indices if cat_indices else None)


@dataclass
class FeatureImportanceResult:
    """Container for feature importance results."""

    feature_names: List[str]
    importances: np.ndarray
    importance_type: str

    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame sorted by importance."""
        return pl.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.importances,
                "importance_type": [self.importance_type] * len(self.feature_names),
            }
        ).sort("importance", descending=True)


def get_feature_importance(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    feature_names: Optional[List[str]] = None,
    importance_type: Literal[
        "PredictionValuesChange", "LossFunctionChange", "FeatureImportance"
    ] = "PredictionValuesChange",
    pool: Optional[Pool] = None,
    config: Optional[Dict[str, Any]] = None,
    df: Optional[pl.DataFrame] = None,
) -> FeatureImportanceResult:
    """Get feature importance from a trained CatBoost model.

    Args:
        model: Trained CatBoost model
        feature_names: List of feature names (optional, will use indices if not provided)
        importance_type: Type of importance to calculate:
            - 'PredictionValuesChange': Average change in prediction when feature is removed
            - 'LossFunctionChange': Average change in loss when feature is removed
            - 'FeatureImportance': Built-in CatBoost feature importance
        pool: CatBoost Pool object (required for PredictionValuesChange and LossFunctionChange)
        config: Optional config dict to extract feature names from
        df: Optional DataFrame to infer feature names from (used with config)

    Returns:
        FeatureImportanceResult object containing feature names and importances

    Example:
        # Using explicit feature names
        importance = get_feature_importance(model, feature_names, importance_type='PredictionValuesChange', pool=train_pool)

        # Using config file
        importance = get_feature_importance(model, config=config, pool=train_pool)

        importance_df = importance.to_polars()
        print(importance_df.head(10))
    """
    # Get feature names from config if not provided
    if feature_names is None and config is not None:
        feature_names = _get_feature_names_from_config(config, df)
    if importance_type == "FeatureImportance":
        importances = model.feature_importances_
    else:
        if pool is None:
            raise ValueError(f"Pool is required for importance_type='{importance_type}'")
        importances = model.get_feature_importance(pool, type=importance_type)

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) does not match number of features ({len(importances)})"
        )

    return FeatureImportanceResult(
        feature_names=feature_names, importances=importances, importance_type=importance_type
    )


def plot_feature_importance(
    importance_result: FeatureImportanceResult,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    backend: Optional[Literal["matplotlib", "plotly"]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Union[plt.Figure, go.Figure]:
    """Plot feature importance as a horizontal bar chart.

    Args:
        importance_result: FeatureImportanceResult from get_feature_importance()
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        title: Optional custom title
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from

    Returns:
        Matplotlib or Plotly figure

    Example:
        importance = get_feature_importance(model, feature_names, pool=train_pool)
        fig = plot_feature_importance(importance, top_n=15, backend='plotly')
        # Or use backend from config
        fig = plot_feature_importance(importance, top_n=15, config=config)
        fig.show()
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)

    # Get top N features
    df = importance_result.to_polars().head(top_n)

    if backend == "plotly":
        # Create plotly figure
        fig = go.Figure()

        # Add horizontal bar chart (reversed so highest is at top)
        fig.add_trace(
            go.Bar(
                x=df["importance"].to_numpy(),
                y=df["feature"].to_list(),
                orientation="h",
                marker=dict(color="steelblue"),
            )
        )

        # Update layout
        plot_title = (
            title if title else f"Top {top_n} Features by {importance_result.importance_type}"
        )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            yaxis=dict(autorange="reversed"),  # Highest at top
            showlegend=False,
        )

        return fig
    else:
        # Matplotlib backend
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bar chart (reversed so highest is at top)
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df["importance"].to_numpy(), align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature"].to_list())
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Top {top_n} Features by {importance_result.importance_type}")

        plt.tight_layout()
        return fig


@dataclass
class ShapResult:
    """Container for SHAP analysis results."""

    shap_values: np.ndarray
    base_value: Union[float, np.ndarray]
    data: np.ndarray
    feature_names: List[str]

    def to_polars(self) -> pl.DataFrame:
        """Convert SHAP values to Polars DataFrame with mean absolute SHAP values per feature."""
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        return pl.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs_shap}).sort(
            "mean_abs_shap", descending=True
        )


def calculate_shap_values(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    pool: Optional[Pool] = None,
    config: Optional[Dict[str, Any]] = None,
    df: Optional[pl.DataFrame] = None,
) -> ShapResult:
    """Calculate SHAP values for a CatBoost model.

    SHAP (SHapley Additive exPlanations) values explain the contribution of each
    feature to the model's predictions for individual samples.

    Args:
        model: Trained CatBoost model
        X: Feature matrix for which to calculate SHAP values
        feature_names: List of feature names (optional)
        pool: CatBoost Pool object (optional, recommended for categorical features)
        config: Optional config dict to extract feature names from
        df: Optional DataFrame to infer feature names from (used with config)

    Returns:
        ShapResult object containing SHAP values and metadata

    Example:
        # Using explicit feature names
        shap_result = calculate_shap_values(model, X_test, feature_names)

        # Using config file
        shap_result = calculate_shap_values(model, X_test, config=config)

        shap_df = shap_result.to_polars()
        print(shap_df.head(10))
    """
    # Get feature names from config if not provided
    if feature_names is None and config is not None:
        feature_names = _get_feature_names_from_config(config, df)
    # Use CatBoost's native SHAP calculation
    if pool is not None:
        shap_values = model.get_feature_importance(pool, type="ShapValues")
    else:
        # Create a temporary pool
        temp_pool = Pool(X)
        shap_values = model.get_feature_importance(temp_pool, type="ShapValues")

    # CatBoost returns SHAP values with base value in last column
    base_value = shap_values[:, -1]
    shap_values = shap_values[:, :-1]

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    if len(feature_names) != shap_values.shape[1]:
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) does not match number of features ({shap_values.shape[1]})"
        )

    return ShapResult(
        shap_values=shap_values, base_value=base_value, data=X, feature_names=feature_names
    )


def plot_shap_summary(
    shap_result: ShapResult,
    plot_type: Literal["dot", "bar", "violin"] = "dot",
    max_display: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a SHAP summary plot.

    Args:
        shap_result: ShapResult from calculate_shap_values()
        plot_type: Type of plot:
            - 'dot': Scatter plot showing SHAP values for each sample
            - 'bar': Bar plot showing mean absolute SHAP values
            - 'violin': Violin plot showing distribution of SHAP values
        max_display: Maximum number of features to display
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        Matplotlib figure

    Example:
        shap_result = calculate_shap_values(model, X_test, feature_names)
        fig = plot_shap_summary(shap_result, plot_type='dot')
        plt.show()
    """
    fig = plt.figure(figsize=figsize)

    if plot_type == "dot":
        shap.summary_plot(
            shap_result.shap_values,
            shap_result.data,
            feature_names=shap_result.feature_names,
            max_display=max_display,
            show=False,
        )
    elif plot_type == "bar":
        shap.summary_plot(
            shap_result.shap_values,
            shap_result.data,
            feature_names=shap_result.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
    elif plot_type == "violin":
        shap.summary_plot(
            shap_result.shap_values,
            shap_result.data,
            feature_names=shap_result.feature_names,
            plot_type="violin",
            max_display=max_display,
            show=False,
        )
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'dot', 'bar', or 'violin'")

    if title:
        plt.title(title)

    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    shap_result: ShapResult,
    sample_index: int = 0,
    max_display: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a SHAP waterfall plot for a single prediction.

    Shows how each feature contributes to pushing the prediction from the base value
    to the final prediction value.

    Args:
        shap_result: ShapResult from calculate_shap_values()
        sample_index: Index of the sample to explain
        max_display: Maximum number of features to display
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        Matplotlib figure

    Example:
        shap_result = calculate_shap_values(model, X_test, feature_names)
        fig = plot_shap_waterfall(shap_result, sample_index=0)
        plt.show()
    """
    fig = plt.figure(figsize=figsize)

    # Create SHAP Explanation object for the sample
    explanation = shap.Explanation(
        values=shap_result.shap_values[sample_index],
        base_values=(
            shap_result.base_value[sample_index]
            if isinstance(shap_result.base_value, np.ndarray)
            else shap_result.base_value
        ),
        data=shap_result.data[sample_index],
        feature_names=shap_result.feature_names,
    )

    shap.waterfall_plot(explanation, max_display=max_display, show=False)

    if title:
        plt.title(title)

    plt.tight_layout()
    return fig


def plot_shap_force(
    shap_result: ShapResult,
    sample_index: int = 0,
    backend: Optional[Literal["matplotlib", "plotly"]] = None,
    config: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (20, 3),
) -> Union[plt.Figure, shap.plots._force.AdditiveForceVisualizer]:
    """Create a SHAP force plot for a single prediction.

    Shows how each feature pushes the prediction higher (red) or lower (blue)
    from the base value.

    Args:
        shap_result: ShapResult from calculate_shap_values()
        sample_index: Index of the sample to explain
        matplotlib: If True, use matplotlib backend; if False, use interactive visualization
        figsize: Figure size (width, height) - only used if matplotlib=True

    Returns:
        Matplotlib figure if matplotlib=True, otherwise interactive visualization

    Example:
        shap_result = calculate_shap_values(model, X_test, feature_names)
        fig = plot_shap_force(shap_result, sample_index=0)
        plt.show()
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)

    base_value = (
        shap_result.base_value[sample_index]
        if isinstance(shap_result.base_value, np.ndarray)
        else shap_result.base_value
    )

    if backend == "matplotlib":
        fig = plt.figure(figsize=figsize)
        shap.force_plot(
            base_value,
            shap_result.shap_values[sample_index],
            shap_result.data[sample_index],
            feature_names=shap_result.feature_names,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        return fig
    else:
        # Return interactive visualization
        return shap.force_plot(
            base_value,
            shap_result.shap_values[sample_index],
            shap_result.data[sample_index],
            feature_names=shap_result.feature_names,
        )


def plot_shap_dependence(
    shap_result: ShapResult,
    feature: Union[str, int],
    interaction_feature: Optional[Union[str, int]] = "auto",
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a SHAP dependence plot showing the effect of a single feature.

    Shows how the SHAP value of a feature varies with its value, optionally
    colored by an interaction feature.

    Args:
        shap_result: ShapResult from calculate_shap_values()
        feature: Feature name or index to plot
        interaction_feature: Feature for coloring (auto-detects strongest interaction by default)
        figsize: Figure size (width, height)
        title: Optional custom title

    Returns:
        Matplotlib figure

    Example:
        shap_result = calculate_shap_values(model, X_test, feature_names)
        fig = plot_shap_dependence(shap_result, 'age')
        plt.show()
    """
    fig = plt.figure(figsize=figsize)

    # Convert feature name to index if needed
    if isinstance(feature, str):
        if feature not in shap_result.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature_names")
        feature_idx = shap_result.feature_names.index(feature)
    else:
        feature_idx = feature
        feature = shap_result.feature_names[feature_idx]

    # Convert interaction feature name to index if needed
    if interaction_feature is not None and interaction_feature != "auto":
        if isinstance(interaction_feature, str):
            if interaction_feature not in shap_result.feature_names:
                raise ValueError(
                    f"Interaction feature '{interaction_feature}' not found in feature_names"
                )
            interaction_feature = shap_result.feature_names.index(interaction_feature)

    shap.dependence_plot(
        feature_idx,
        shap_result.shap_values,
        shap_result.data,
        feature_names=shap_result.feature_names,
        interaction_index=interaction_feature,
        show=False,
    )

    if title:
        plt.title(title)
    else:
        plt.title(f"SHAP Dependence Plot: {feature}")

    plt.tight_layout()
    return fig


@dataclass
class PartialDependenceResult:
    """Container for partial dependence results."""

    feature_name: str
    feature_values: np.ndarray
    pd_values: np.ndarray
    is_categorical: bool

    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        return pl.DataFrame(
            {"feature_value": self.feature_values, "partial_dependence": self.pd_values}
        )


def calculate_partial_dependence(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    X: np.ndarray,
    feature_index: int,
    feature_name: Optional[str] = None,
    grid_resolution: int = 50,
    percentile_range: Tuple[float, float] = (0.05, 0.95),
) -> PartialDependenceResult:
    """Calculate partial dependence for a single feature.

    Partial dependence shows the marginal effect of a feature on the predicted outcome,
    averaging out the effects of all other features.

    Args:
        model: Trained CatBoost model
        X: Feature matrix (used to determine feature range)
        feature_index: Index of the feature to analyze
        feature_name: Name of the feature (optional)
        grid_resolution: Number of points to evaluate
        percentile_range: Range of feature values to consider (as percentiles)

    Returns:
        PartialDependenceResult object

    Example:
        pd_result = calculate_partial_dependence(model, X_test, feature_index=0, feature_name='age')
        pd_df = pd_result.to_polars()
        print(pd_df)
    """
    if feature_name is None:
        feature_name = f"feature_{feature_index}"

    # Get feature values
    feature_values = X[:, feature_index]

    # Check if categorical (assuming categorical if unique values < 20)
    unique_values = np.unique(feature_values)
    is_categorical = len(unique_values) < 20

    if is_categorical:
        # Use actual unique values for categorical features
        grid_values = unique_values
    else:
        # Create grid for continuous features
        min_val = np.percentile(feature_values, percentile_range[0] * 100)
        max_val = np.percentile(feature_values, percentile_range[1] * 100)
        grid_values = np.linspace(min_val, max_val, grid_resolution)

    # Calculate partial dependence
    pd_values = []
    for grid_val in grid_values:
        # Create modified dataset with feature set to grid value
        X_modified = X.copy()
        X_modified[:, feature_index] = grid_val

        # Get predictions and average
        predictions = model.predict(X_modified)
        pd_values.append(predictions.mean())

    pd_values = np.array(pd_values)

    return PartialDependenceResult(
        feature_name=feature_name,
        feature_values=grid_values,
        pd_values=pd_values,
        is_categorical=is_categorical,
    )


def plot_partial_dependence(
    pd_result: PartialDependenceResult,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    backend: Optional[Literal["matplotlib", "plotly"]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Union[plt.Figure, go.Figure]:
    """Plot partial dependence for a single feature.

    Args:
        pd_result: PartialDependenceResult from calculate_partial_dependence()
        figsize: Figure size (width, height)
        title: Optional custom title
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from

    Returns:
        Matplotlib or Plotly figure

    Example:
        pd_result = calculate_partial_dependence(model, X_test, feature_index=0, feature_name='age')
        fig = plot_partial_dependence(pd_result, backend='plotly')
        # Or use backend from config
        fig = plot_partial_dependence(pd_result, config=config)
        fig.show()
    """
    # Determine backend: explicit parameter > config > default
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)

    if backend == "plotly":
        # Create plotly figure
        fig = go.Figure()

        if pd_result.is_categorical:
            # Bar plot for categorical features
            fig.add_trace(
                go.Bar(
                    x=[str(v) for v in pd_result.feature_values],
                    y=pd_result.pd_values,
                    marker=dict(color="steelblue"),
                )
            )
        else:
            # Line plot for continuous features
            fig.add_trace(
                go.Scatter(
                    x=pd_result.feature_values,
                    y=pd_result.pd_values,
                    mode="lines",
                    line=dict(width=2, color="steelblue"),
                )
            )

        # Update layout
        plot_title = title if title else f"Partial Dependence: {pd_result.feature_name}"
        fig.update_layout(
            title=plot_title,
            xaxis_title=pd_result.feature_name,
            yaxis_title="Partial Dependence",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            showlegend=False,
        )

        return fig
    else:
        # Matplotlib backend
        fig, ax = plt.subplots(figsize=figsize)

        if pd_result.is_categorical:
            # Bar plot for categorical features
            ax.bar(range(len(pd_result.feature_values)), pd_result.pd_values)
            ax.set_xticks(range(len(pd_result.feature_values)))
            ax.set_xticklabels(pd_result.feature_values, rotation=45, ha="right")
        else:
            # Line plot for continuous features
            ax.plot(pd_result.feature_values, pd_result.pd_values, linewidth=2)

        ax.set_xlabel(pd_result.feature_name)
        ax.set_ylabel("Partial Dependence")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Partial Dependence: {pd_result.feature_name}")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def plot_multiple_partial_dependence(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    X: np.ndarray,
    feature_indices: List[int],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 50,
    figsize: Tuple[int, int] = (15, 10),
    title: Optional[str] = None,
    backend: Optional[Literal["matplotlib", "plotly"]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Union[plt.Figure, go.Figure]:
    """Plot partial dependence for multiple features in a grid.

    Args:
        model: Trained CatBoost model
        X: Feature matrix
        feature_indices: List of feature indices to plot
        feature_names: List of all feature names (optional)
        grid_resolution: Number of points to evaluate for continuous features
        figsize: Figure size (width, height)
        title: Optional custom title
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
        config: Optional config dict to extract backend from

    Returns:
        Matplotlib or Plotly figure with subplots

    Example:
        fig = plot_multiple_partial_dependence(
            model, X_test,
            feature_indices=[0, 1, 2, 3],
            feature_names=feature_names,
            backend='plotly'
        )
        # Or use backend from config
        fig = plot_multiple_partial_dependence(
            model, X_test,
            feature_indices=[0, 1, 2, 3],
            feature_names=feature_names,
            config=config
        )
        fig.show()
    """
    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)

    n_features = len(feature_indices)

    if backend == "plotly":
        # Create plotly figure with subplots
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[
                feature_names[idx] if feature_names else f"feature_{idx}" for idx in feature_indices
            ],
        )

        for idx, feature_idx in enumerate(feature_indices):
            feature_name = feature_names[feature_idx] if feature_names else f"feature_{feature_idx}"

            # Calculate partial dependence
            pd_result = calculate_partial_dependence(
                model, X, feature_idx, feature_name, grid_resolution
            )

            # Determine subplot position
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            # Add trace
            if pd_result.is_categorical:
                fig.add_trace(
                    go.Bar(
                        x=[str(v) for v in pd_result.feature_values],
                        y=pd_result.pd_values,
                        showlegend=False,
                        marker=dict(color="steelblue"),
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=pd_result.feature_values,
                        y=pd_result.pd_values,
                        mode="lines",
                        showlegend=False,
                        line=dict(width=2, color="steelblue"),
                    ),
                    row=row,
                    col=col,
                )

            # Update axes labels
            fig.update_xaxes(title_text=feature_name, row=row, col=col)
            fig.update_yaxes(title_text="PD", row=row, col=col)

        # Update layout
        plot_title = title if title else "Partial Dependence Plots"
        fig.update_layout(
            title_text=plot_title, width=figsize[0] * 100, height=figsize[1] * 100, showlegend=False
        )

        return fig
    else:
        # Matplotlib backend
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, feature_idx in enumerate(feature_indices):
            feature_name = feature_names[feature_idx] if feature_names else f"feature_{feature_idx}"

            # Calculate partial dependence
            pd_result = calculate_partial_dependence(
                model, X, feature_idx, feature_name, grid_resolution
            )

            # Plot on subplot
            ax = axes[idx]
            if pd_result.is_categorical:
                ax.bar(range(len(pd_result.feature_values)), pd_result.pd_values)
                ax.set_xticks(range(len(pd_result.feature_values)))
                ax.set_xticklabels(pd_result.feature_values, rotation=45, ha="right")
            else:
                ax.plot(pd_result.feature_values, pd_result.pd_values, linewidth=2)

            ax.set_xlabel(feature_name)
            ax.set_ylabel("Partial Dependence")
            ax.set_title(f"PD: {feature_name}")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=14, y=1.00)

        plt.tight_layout()
        return fig


def create_evaluation_report(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    pool: Optional[Pool] = None,
    output_dir: Optional[str] = None,
    top_n_features: int = 10,
    config: Optional[Dict[str, Any]] = None,
    df: Optional[pl.DataFrame] = None,
    backend: Optional[Literal["matplotlib", "plotly"]] = None,
) -> dict:
    """Create a comprehensive evaluation report with all interpretation methods.

    Generates feature importance, SHAP analysis, and partial dependence plots,
    optionally saving them to disk.

    Args:
        model: Trained CatBoost model
        X: Feature matrix
        y: Target values
        feature_names: List of feature names (optional if config provided)
        pool: CatBoost Pool object (optional but recommended)
        output_dir: Directory to save plots (optional)
        top_n_features: Number of top features to analyze in detail
        config: Optional config dict to extract feature names and backend from
        df: Optional DataFrame to infer feature names from (used with config)
        backend: Plotting backend to use ("matplotlib" or "plotly"). If None, uses config or defaults to "matplotlib"
            Note: SHAP plots always use matplotlib regardless of backend setting

    Returns:
        Dictionary containing all results and figures

    Example:
        # Using explicit feature names with matplotlib
        report = create_evaluation_report(
            model, X_test, y_test, feature_names,
            pool=test_pool,
            output_dir='./evaluation_output'
        )

        # Using config file with plotly backend specified
        report = create_evaluation_report(
            model, X_test, y_test,
            config=config,
            pool=test_pool,
            output_dir='./evaluation_output',
            backend='plotly'
        )

        # Using config file with backend from config
        # (config contains "plotting_backend": "plotly")
        report = create_evaluation_report(
            model, X_test, y_test,
            config=config,
            pool=test_pool,
            output_dir='./evaluation_output'
        )
    """
    # Get feature names from config if not provided
    if feature_names is None and config is not None:
        feature_names = _get_feature_names_from_config(config, df)

    # Determine backend from config if not explicitly provided
    backend = _get_backend_from_config(backend, config)

    if feature_names is None:
        raise ValueError("feature_names must be provided either directly or via config")

    results = {}

    # 1. Feature Importance
    print("Calculating feature importance...")
    importance = get_feature_importance(model, feature_names, pool=pool)
    results["feature_importance"] = importance
    results["feature_importance_df"] = importance.to_polars()

    fig_importance = plot_feature_importance(importance, top_n=top_n_features, backend=backend)
    results["fig_feature_importance"] = fig_importance

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if backend == "matplotlib":
            fig_importance.savefig(
                f"{output_dir}/feature_importance.png", dpi=300, bbox_inches="tight"
            )
        else:
            fig_importance.write_html(f"{output_dir}/feature_importance.html")

    # 2. SHAP Analysis (matplotlib only for SHAP plots)
    print("Calculating SHAP values...")
    shap_result = calculate_shap_values(model, X, feature_names, pool)
    results["shap_result"] = shap_result
    results["shap_df"] = shap_result.to_polars()

    # SHAP summary plot
    fig_shap_summary = plot_shap_summary(shap_result, plot_type="dot", max_display=top_n_features)
    results["fig_shap_summary"] = fig_shap_summary

    if output_dir:
        fig_shap_summary.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches="tight")

    # SHAP bar plot
    fig_shap_bar = plot_shap_summary(shap_result, plot_type="bar", max_display=top_n_features)
    results["fig_shap_bar"] = fig_shap_bar

    if output_dir:
        fig_shap_bar.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches="tight")

    # 3. Partial Dependence for top features
    print("Calculating partial dependence...")
    top_feature_indices = importance.to_polars().head(top_n_features)["feature"].to_list()
    top_feature_indices = [feature_names.index(f) for f in top_feature_indices]

    fig_pd = plot_multiple_partial_dependence(
        model,
        X,
        top_feature_indices,
        feature_names,
        title=f"Partial Dependence: Top {top_n_features} Features",
        backend=backend,
    )
    results["fig_partial_dependence"] = fig_pd

    if output_dir:
        if backend == "matplotlib":
            fig_pd.savefig(f"{output_dir}/partial_dependence.png", dpi=300, bbox_inches="tight")
        else:
            fig_pd.write_html(f"{output_dir}/partial_dependence.html")

    print(
        f"Evaluation report complete. Generated {len([k for k in results.keys() if k.startswith('fig_')])} figures."
    )

    return results


# =============================================================================
# Usage Examples with Config File
# =============================================================================
"""
USAGE EXAMPLES WITH CONFIG FILE:

Config file structure (config.json):
    ```json
    {
        "model_type": "classifier",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2", "feature3"],
        "categorical_features": ["feature1"],
        "plotting_backend": "plotly",
        "model_params": {
            "learning_params": {
                "iterations": 1000,
                "learning_rate": 0.1
            }
        }
    }
    ```

Note: The "plotting_backend" field in config can be "matplotlib" or "plotly".
      If not specified, defaults to "matplotlib".

1. Basic usage with config file:
    ```python
    from ml_store import ml_functions, ml_evaluation

    # Load config
    config = ml_functions.load_config('config.json')

    # Load model
    model = ml_functions.load_model('model.cbm', model_type=config['model_type'])

    # Prepare test data
    X_test, y_test, feature_names, cat_indices, _, _ = ml_functions.create_modelling_data(
        test_df, config, train_ratio=0.0, test_ratio=1.0
    )

    # Create pool from config
    test_pool = ml_evaluation.create_pool_from_config(X_test, y_test, config)

    # Get feature importance using config
    importance = ml_evaluation.get_feature_importance(
        model,
        config=config,
        pool=test_pool
    )
    print(importance.to_polars().head(10))
    ```

2. SHAP analysis with config:
    ```python
    # Calculate SHAP values using config
    shap_result = ml_evaluation.calculate_shap_values(
        model,
        X_test,
        config=config,
        pool=test_pool
    )

    # Plot SHAP summary
    fig = ml_evaluation.plot_shap_summary(shap_result, plot_type='dot')
    plt.show()

    # Plot SHAP dependence for a specific feature
    fig = ml_evaluation.plot_shap_dependence(shap_result, 'age')
    plt.show()
    ```

3. Comprehensive evaluation report with config:
    ```python
    # Generate full report using config
    # If config contains "plotting_backend": "plotly", all plots will use plotly
    report = ml_evaluation.create_evaluation_report(
        model,
        X_test,
        y_test,
        config=config,  # Will use plotting_backend from config
        pool=test_pool,
        output_dir='./evaluation_output',
        top_n_features=15
    )

    # Access results
    print(report['feature_importance_df'])
    print(report['shap_df'])

    # Display figures (plotly figures have .show() method)
    report['fig_feature_importance'].show()
    report['fig_shap_summary'].show()
    ```

4. Using plotting backend from config:
    ```python
    # Config contains "plotting_backend": "plotly"
    # All plotting functions will automatically use plotly

    importance = ml_evaluation.get_feature_importance(
        model,
        config=config,  # Backend read from config
        pool=test_pool
    )
    fig = ml_evaluation.plot_feature_importance(importance, config=config)
    fig.show()  # Interactive plotly plot

    # Override config backend with explicit parameter
    fig_matplotlib = ml_evaluation.plot_feature_importance(
        importance,
        backend='matplotlib',  # Explicit override
        config=config
    )
    plt.show()
    ```

5. Without config (traditional approach still works):
    ```python
    # Explicit feature names
    feature_names = ['age', 'income', 'credit_score']

    importance = ml_evaluation.get_feature_importance(
        model,
        feature_names=feature_names,
        pool=test_pool
    )

    shap_result = ml_evaluation.calculate_shap_values(
        model,
        X_test,
        feature_names=feature_names,
        pool=test_pool
    )
    ```

5. Partial dependence with config:
    ```python
    # Get feature names from config
    feature_names = config.get('feature_columns', [])

    # Calculate PD for specific feature
    pd_result = ml_evaluation.calculate_partial_dependence(
        model,
        X_test,
        feature_index=0,
        feature_name=feature_names[0]
    )

    # Plot
    fig = ml_evaluation.plot_partial_dependence(pd_result)
    plt.show()
    ```
"""
