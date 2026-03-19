import os
import pandas as pd

from data_loader import load_data

from eda import run_eda

from preprocessing import (
    add_engineered_features,
    split_features_target,
    train_test_split_data,
)

from models import (
    get_models,
    get_cv,
    run_cross_validation,
    fit_model,
    tune_xgboost,
)

from evaluation import (
    evaluate_regression_model,
    evaluate_multiple_models,
)

from plots import (
    plot_target_distribution,
    plot_feature_histograms,
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_residuals_boxplot,
    plot_feature_importance,
)

from config import (
    RANDOM_SEED,
    TEST_SIZE,
    N_SPLITS_CROSS_V,
    TARGET_COLUMN,
    TABLES_DIR,
)



def main():
    # Load data
    df = load_data()

    # EDA (textual with no plots)
    run_eda(df)

    # EDA (plots)
    plot_target_distribution(df, TARGET_COLUMN)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)

    # Feature engineering
    df_fe = add_engineered_features(df)

    # Split features / target
    X, y = split_features_target(df, TARGET_COLUMN)
    X_fe, y_fe =  split_features_target(df_fe, TARGET_COLUMN)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split_data(
        X_fe, y_fe, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Build models
    models = get_models(random_state=RANDOM_SEED)
    models_fe = get_models(random_state=RANDOM_SEED)

    # Cross-validation
    cross_validation = get_cv(n_splits=N_SPLITS_CROSS_V, random_state=RANDOM_SEED)
    
    cv_results_list = []

    print("\n=== CROSS-VALIDATION RESULTS - ORIGINAL FEATURES ===")
    for name, model in models.items():
        cross_validation_results = run_cross_validation(model, X, y, cross_validation)
        mean_rmse = -cross_validation_results["test_rmse"].mean()
        mean_mae = -cross_validation_results["test_mae"].mean()
        mean_r2 = cross_validation_results["test_r2"].mean()

        print(f"\n{name}")
        print(f"CV RMSE: {mean_rmse:.3f}")
        print(f"CV MAE:  {mean_mae:.3f}")
        print(f"CV R2:   {mean_r2:.3f}")
        
        cv_results_list.append({
            "Model": f"{name} - Original",
            "CV_RMSE": mean_rmse,
            "CV_MAE": mean_mae,
            "CV_R2": mean_r2
        })

    print("\n=== CROSS-VALIDATION RESULTS - FEATURE ENGINEERED ===")
    for name, model in models_fe.items():
        cross_validation_results = run_cross_validation(model, X_fe, y_fe, cross_validation)
        mean_rmse = -cross_validation_results["test_rmse"].mean()
        mean_mae = -cross_validation_results["test_mae"].mean()
        mean_r2 = cross_validation_results["test_r2"].mean()

        print(f"\n{name}")
        print(f"CV RMSE: {mean_rmse:.3f}")
        print(f"CV MAE:  {mean_mae:.3f}")
        print(f"CV R2:   {mean_r2:.3f}")
        
        cv_results_list.append({
            "Model": f"{name} - FE",
            "CV_RMSE": mean_rmse,
            "CV_MAE": mean_mae,
            "CV_R2": mean_r2
        })
        
    cv_results_df = pd.DataFrame(cv_results_list)
    cv_results_df.to_csv(os.path.join(TABLES_DIR, "cross_validation_results.csv"), index=False)

    # Fit baseline models - original features
    fitted_models = {}
    for name, model in models.items():
        fitted_models[name] = fit_model(model, X_train, y_train)

    # Fit baseline models - feature engineered
    fitted_models_fe = {}
    for name, model in models_fe.items():
        fitted_models_fe[name] = fit_model(model, X_train_fe, y_train_fe) 

    # Hyperparameter tuning XGBoost
    print("\n=== XGBOOST TUNING ===")
    xgb_search = tune_xgboost(X_train_fe, y_train_fe, random_state=RANDOM_SEED)
    best_xgb = xgb_search.best_estimator_
    fitted_models_fe["XGBoost Tuned"] = best_xgb
    print("Best XGBoost parameters:")
    print(xgb_search.best_params_)
    
    # Final evaluation on test set
    combined_results = {}

    for name, model in fitted_models.items():
        metrics = evaluate_regression_model(model, X_test, y_test)
        combined_results[f"{name} - Original"] = metrics

    for name, model in fitted_models_fe.items():
        metrics = evaluate_regression_model(model, X_test_fe, y_test_fe)
        combined_results[f"{name} - FE"] = metrics

    results_df = evaluate_multiple_models(combined_results)

    print("\n=== TEST SET RESULTS - ALL MODELS ===")
    print(results_df)
    
    results_df.to_csv(os.path.join(TABLES_DIR, "test_set_results.csv"))
    
    # Final plots - original features
    residuals_original = {}
    
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)

        plot_actual_vs_predicted(y_test, y_pred, model_name=f"{name} - Original")
        plot_residual_distribution(y_test, y_pred, model_name=f"{name} - Original")

        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                model.feature_importances_,
                X_test.columns,
                model_name=f"{name} - Original"
            )
        elif hasattr(model, "named_steps"):
            final_estimator = model.named_steps.get("model", None)
            if final_estimator is not None and hasattr(final_estimator, "feature_importances_"):
                plot_feature_importance(
                    final_estimator.feature_importances_,
                    X_test.columns,
                    model_name=f"{name} - Original"
                )
          
        residuals_original[f"{name} - Original"] = y_test - y_pred
    
    plot_residuals_boxplot(residuals_original)

    # Final plots - feature engineered
    residuals_fe = {}
    
    for name, model in fitted_models_fe.items():
        y_pred = model.predict(X_test_fe)

        plot_actual_vs_predicted(y_test_fe, y_pred, model_name=f"{name} - FE")
        plot_residual_distribution(y_test_fe, y_pred, model_name=f"{name} - FE")

        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                model.feature_importances_,
                X_test_fe.columns,
                model_name=f"{name} - FE"
            )
        elif hasattr(model, "named_steps"):
            final_estimator = model.named_steps.get("model", None)
            if final_estimator is not None and hasattr(final_estimator, "feature_importances_"):
                plot_feature_importance(
                    final_estimator.feature_importances_,
                    X_test_fe.columns,
                    model_name=f"{name} - FE"
                )
                
        residuals_fe[f"{name} - FE"] = y_test_fe - y_pred
            
    plot_residuals_boxplot(residuals_fe)



if __name__ == "__main__":
    main()