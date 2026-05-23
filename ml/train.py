import pandas as pd
import numpy as np
import pickle
import logging
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "square_meters", "bedrooms", "bathrooms", "year_built",
    "has_elevator", "subway_distance_m", "school_distance",
    "west_lake_distance_m", "neighborhood_amenities_score",
]

CATEGORICAL_FEATURES = [
    "property_type", "district", "floor_level",
    "decoration", "parking_rate", "subway_line",
]

TARGET = "price"
MODEL_PATH = "xgb_pipeline_v2.pkl"


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["subway_distance_m"] = df["subway_distance_m"].fillna(9999)
    return df


def build_pipeline() -> Pipeline:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb)])


def print_feature_importance(pipeline: Pipeline) -> None:
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = cat_feature_names + NUMERIC_FEATURES

    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("\n--- Feature Importances ---")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        logger.info(f"  {row['feature']:<45} {row['importance']:.4f}  {bar}")


def train(dataset_path: str = "hangzhou_housing_dataset.csv") -> None:
    logger.info("Loading dataset: %s", dataset_path)
    df = load_and_preprocess(dataset_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    logger.info("Training XGBoost pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("MAE: %.2f | R²: %.4f", mae, r2)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Pipeline saved to %s", MODEL_PATH)

    print_feature_importance(pipeline)


if __name__ == "__main__":
    train()

    