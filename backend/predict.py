import logging
import pickle
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path("../ml/XGmodel.pkl")
MODEL_VERSION = "v2.0.0"

NUMERIC_FEATURES = [
    "area_sqm", "bedrooms", "living_rooms", "bathrooms",
    "floor", "total_floors", "has_elevator", "building_age_years",
    "subway_distance_m", "west_lake_distance_m", "green_ratio",
    "parking_ratio", "management_fee_yuan_per_sqm",
]

CATEGORICAL_FEATURES = [
    "district", "orientation", "decoration",
    "property_type", "developer", "subway_line", "school_district_tier",
]

DISTRICT_ENRICHMENT: dict[str, dict[str, float]] = {
    "Xihu":       {"west_lake_distance_m": 500.0,   "green_ratio": 0.42},
    "Binjiang":   {"west_lake_distance_m": 8200.0,  "green_ratio": 0.35},
    "Gongshu":    {"west_lake_distance_m": 5100.0,  "green_ratio": 0.28},
    "Shangcheng": {"west_lake_distance_m": 3400.0,  "green_ratio": 0.31},
    "Xiaoshan":   {"west_lake_distance_m": 18000.0, "green_ratio": 0.25},
    "Yuhang":     {"west_lake_distance_m": 22000.0, "green_ratio": 0.22},
}

# Fallback values used when an optional field is omitted by the caller.
# Numeric defaults are rough Hangzhou-market medians; categorical defaults
# are a neutral "Unknown" bucket the pipeline's encoder should recognize.
OPTIONAL_DEFAULTS: dict[str, object] = {
    "living_rooms": 1,
    "orientation": "Unknown",
    "property_type": "Unknown",
    "developer": "Unknown",
    "subway_line": "Unknown",
    "subway_distance_m": 9999.0,
    "school_district_tier": "Unknown",
    "parking_ratio": 1.0,
    "management_fee_yuan_per_sqm": 3.0,
}


class HousePredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model artifact not found at '{MODEL_PATH}'. "
                "Run train.py before starting the server."
            )
        with open(MODEL_PATH, "rb") as f:
            self._pipeline = pickle.load(f)
        logger.info("Loaded pipeline from %s (version=%s)", MODEL_PATH, MODEL_VERSION)

    def _enrich(self, payload: dict) -> dict:
        district = payload["district"]
        enrichment = DISTRICT_ENRICHMENT[district]
        payload["west_lake_distance_m"] = enrichment["west_lake_distance_m"]
        payload["green_ratio"] = enrichment["green_ratio"]
        return payload

    def _fill_optional_defaults(self, payload: dict) -> dict:
        for field, default_value in OPTIONAL_DEFAULTS.items():
            if payload.get(field) is None:
                payload[field] = default_value
        return payload

    def predict(self, payload: dict) -> tuple[float, float]:
        logger.info("Incoming request payload: %s", payload)

        payload = self._fill_optional_defaults(payload)
        payload = self._enrich(payload)

        df = pd.DataFrame([payload])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        t0 = time.perf_counter()
        price = float(self._pipeline.predict(df)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info("Inference complete | predicted_price=%.2f 万 | latency=%.3f ms", price, latency_ms)
        return price, latency_ms