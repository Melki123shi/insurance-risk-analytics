from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import xgboost
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import sklearn.tree
import sklearn.ensemble


def load_models():
    lr_model = joblib.load("../models/lr_model.pkl")
    dt_model = joblib.load("../models/dt_model.pkl")
    rfr_model = joblib.load("../models/rfr_model.pkl")
    xgb_model = joblib.load("../models/xgb_model.pkl")
    return lr_model, dt_model, rfr_model, xgb_model


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def label_encode_columns(df: pd.DataFrame, categorical_columns):
    """
    Label encode the specified categorical columns.

    Parameters:
    - df: Input DataFrame
    - categorical_columns: List of column names to encode

    Returns:
    - Encoded DataFrame
    - Dictionary of fitted LabelEncoders keyed by column name
    """
    df = df.copy()
    encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            values = df[col].fillna("missing").astype(str)
            df[col] = encoder.fit_transform(values)
            encoders[col] = encoder

    return df, encoders


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features relevant to TotalPremium prediction.
    """
    df = df.copy()

    # =============================================
    # 1. Vehicle Age (Very Important Risk Factor)
    # =============================================
    if "RegistrationYear" in df.columns:
        # Handle weird format like: "1970-01-01 00:00:00.000002004"
        df["RegistrationYear"] = (
            pd.to_datetime(df["RegistrationYear"], errors="coerce").dt.year
        )

        df["VehicleAge"] = 2026 - df["RegistrationYear"]
        df["VehicleAge"] = df["VehicleAge"].clip(lower=0)
        print("✓ Created VehicleAge")

    # =============================================
    # 2. Transaction / Policy Time Features
    # =============================================
    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"], errors="coerce"
        )
        df["PolicyMonth"] = df["TransactionMonth"].dt.month
        df["PolicyYear"] = df["TransactionMonth"].dt.year
        df["PolicyDay"] = df["TransactionMonth"].dt.day
        df["IsWeekendTransaction"] = (
            df["TransactionMonth"].dt.weekday >= 5
        )
        msg = (
            "✓ Created PolicyMonth, PolicyYear, PolicyDay, "
            "IsWeekendTransaction"
        )
        print(msg)

    # =============================================
    # 3. Vehicle & Engine Features
    # =============================================
    if "cubiccapacity" in df.columns and "kilowatts" in df.columns:
        df["EngineCC_Per_KW"] = df["cubiccapacity"] / (
            df["kilowatts"] + 1
        )  # avoid div by zero
        df["KW_Per_CC"] = df["kilowatts"] / (df["cubiccapacity"] + 1)
        print("✓ Created EngineCC_Per_KW and KW_Per_CC")

    if "Cylinders" in df.columns:
        df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")

    # =============================================
    # 4. Insurance & Cover Features
    # =============================================
    if "SumInsured" in df.columns:
        # log transform (good for skewed data)
        df["SumInsured_Log"] = np.log1p(df["SumInsured"])
        print("✓ Created SumInsured_Log")

    if "ExcessSelected" in df.columns:
        df["ExcessSelected"] = pd.to_numeric(
            df["ExcessSelected"], errors="coerce"
        )
        df["SumInsured_To_Excess"] = (
            df["SumInsured"] / (df["ExcessSelected"] + 1)
        )
        print("✓ Created SumInsured_To_Excess")

    # =============================================
    # 5. Vehicle Make & Model Grouping
    # =============================================
    if "make" in df.columns:
        df["MakeGroup"] = df["make"].str.upper().str.strip()
        # Popular luxury/ high-risk makes
        luxury_makes = [
            "MERCEDES-BENZ",
            "BMW",
            "AUDI",
            "PORSCHE",
            "LAND ROVER",
            "JAGUAR",
        ]
        df["IsLuxuryMake"] = df["MakeGroup"].isin(luxury_makes).astype(int)
        print("✓ Created MakeGroup and IsLuxuryMake")

    if "Model" in df.columns and "make" in df.columns:
        df["Make_Model"] = (
            df["MakeGroup"]
            + "_"
            + df["Model"].astype(str).str.upper().str.strip()
        )
        print("✓ Created Make_Model interaction")

    # =============================================
    # 6. Binary / Categorical Flags
    # =============================================
    binary_cols = [
        "AlarmImmobiliser",
        "TrackingDevice",
        "NewVehicle",
        "IsVATRegistered",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .map({"Yes": 1, "No": 0, True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )
            msg = f"✓ Converted {col} to binary"
            print(msg)

    # =============================================
    # AVOID LEAKAGE: Do NOT use these for TotalPremium prediction
    # =============================================
    # Removed:
    # - SumInsuredPerClaim (uses TotalClaims)
    # - PremiumToSumInsured (uses CalculatedPremiumPerTerm)

    print(f"✅ Feature engineering completed. New shape: {df.shape}")
    return df


def train_models(X_train, y_train):
    """
    Train models with robust NaN handling.
    """
    X_train = X_train.copy()
    y_train = y_train.copy()

    nan_count = X_train.isna().sum().sum()
    print(f"Before cleaning - NaNs in X_train: {nan_count:,}")

    # 1. Fill numeric columns with median
    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        median_val = X_train[col].median()
        if pd.isna(median_val):  # if all values are NaN
            median_val = 0
        X_train[col] = X_train[col].fillna(median_val)

    # 2. Fill object/string columns
    object_cols = X_train.select_dtypes(include=["object"]).columns
    X_train[object_cols] = X_train[object_cols].fillna("missing")

    # 3. Handle remaining non-numeric issues (e.g., bool after encoding)
    for col in X_train.columns:
        if X_train[col].dtype == "bool":
            X_train[col] = X_train[col].astype(int)
        if X_train[col].isna().any():
            if X_train[col].dtype.kind in "biufc":  # numeric
                X_train[col] = X_train[col].fillna(0)
            else:
                X_train[col] = X_train[col].fillna("missing")

    # 4. Clean target
    nan_count_y = y_train.isna().sum()
    if nan_count_y > 0:
        print(f"Removing {nan_count_y} rows with NaN in target")
        mask = ~y_train.isna()
        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask].reset_index(drop=True)

    print(f"✅ Final NaNs in X_train: {X_train.isna().sum().sum():,}")
    print(f"Training with {len(X_train):,} samples")

    # Models
    lr_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )

    # For tree models, they handle NaNs better, but we still clean
    dt_model = DecisionTreeRegressor(random_state=42)
    rfr_model = RandomForestRegressor(random_state=42, n_jobs=-1)  # faster
    xgb_model = xgboost.XGBRegressor(random_state=42, n_jobs=-1)

    # Fit models
    lr_model = lr_pipeline.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rfr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # save the models to ../models directory
    joblib.dump(lr_model, "../models/lr_model.pkl")
    joblib.dump(dt_model, "../models/dt_model.pkl")
    joblib.dump(rfr_model, "../models/rfr_model.pkl")
    joblib.dump(xgb_model, "../models/xgb_model.pkl")

    return lr_model, dt_model, rfr_model, xgb_model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, y_pred


def summery_table(results):
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.round(4))

    # Optional: Save results
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nResults saved to 'model_comparison_results.csv'")


def get_feature_importance(model, X_train, feature_names=None, top_n=15):
    """Get built-in feature importance (for tree models)"""
    if feature_names is None:
        feature_names = X_train.columns.tolist()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = (
            pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            )
            .sort_values("Importance", ascending=False)
        )
        return importance_df.head(top_n)
    else:
        print("Model does not support built-in feature importance.")
        return None


def explain_with_shap(
    model, X_train, X_test, feature_names=None, max_samples=500
):
    """
    Generate SHAP explanations for the model.
    Returns SHAP values and creates summary plots.
    """
    if feature_names is None:
        feature_names = X_train.columns.tolist()

    print("Computing SHAP values... (this may take a moment)")

    # Use TreeExplainer for tree-based models (much faster)
    if isinstance(
        model,
        (
            xgboost.XGBRegressor,
            sklearn.ensemble.RandomForestRegressor,
            sklearn.tree.DecisionTreeRegressor,
        ),
    ):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_train)

    # Use a sample for speed
    X_test_sample = X_test.iloc[:max_samples]

    shap_values = explainer.shap_values(X_test_sample)

    # For regression, shap_values is already the right shape
    if isinstance(shap_values, list):  # some models return list
        shap_values = shap_values[0]

    print("✅ SHAP values computed successfully!")

    # Plot 1: Global Summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        "SHAP Summary Plot - Feature Impact on TotalPremium"
    )
    plt.tight_layout()
    plt.savefig(
        "shap_summary_plot.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Plot 2: Top Features Bar
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("Top Features by Average SHAP Impact")
    plt.tight_layout()
    plt.savefig(
        "shap_top_features.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return shap_values, explainer


def interpret_top_features(shap_values, X_test, feature_names=None, top_n=10):
    """
    Generate business-friendly interpretation of top features.
    """
    if feature_names is None:
        feature_names = X_test.columns.tolist()

    # Get mean absolute SHAP value per feature
    shap_importance = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("Mean_Abs_SHAP", ascending=False)
    )

    print("\n" + "=" * 80)
    print("TOP 10 MOST INFLUENTIAL FEATURES FOR TotalPremium")
    print("=" * 80)

    for i, row in shap_importance.head(top_n).iterrows():
        feature = row["Feature"]
        impact = row["Mean_Abs_SHAP"]
        impact_str = f"→ Avg Impact: {impact:.4f}"
        print(f"{i + 1:2d}. {feature:35} {impact_str}")

    return shap_importance.head(top_n)
