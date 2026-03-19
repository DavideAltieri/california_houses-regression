import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_target_distribution(df: pd.DataFrame, target_column: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[target_column], bins=30, kde=True)
    plt.title("Target Distribution", fontweight="bold")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"), bbox_inches="tight")
    plt.show()


def plot_feature_histograms(df: pd.DataFrame) -> None:
    df.hist(figsize=(12, 7), bins=30)
    plt.suptitle("Features Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "features_distribution.png"), bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_matrix.png"), bbox_inches="tight")
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, model_name: str = "Model") -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Actual vs Predicted", fontweight="bold")

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    filename = f"{model_name}_actual_vs_predicted.png".replace(" ", "_")
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight")
    plt.show()


def plot_residual_distribution(y_true, y_pred, model_name: str = "Model") -> None:
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f"{model_name} - Residual Distribution", fontweight="bold")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    filename = f"{model_name}_residual_distribution.png".replace(" ", "_")
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight")
    plt.show()


def plot_residuals_boxplot(residuals_dict: dict) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=pd.DataFrame(residuals_dict))
    plt.title("Residuals by Model", fontweight="bold")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_by_model.png"), bbox_inches="tight")
    plt.show()


def plot_feature_importance(importances, feature_names, model_name: str = "Model") -> None:
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title(f"{model_name} - Feature Importance", fontweight="bold")
    plt.tight_layout()
    filename = f"{model_name}_feature_importance.png".replace(" ", "_")
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight")
    plt.show()