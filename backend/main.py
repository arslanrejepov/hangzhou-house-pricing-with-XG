import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from schemas import HouseInput, PredictionResponse
from predict import HousePredictor, MODEL_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

predictor: HousePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = HousePredictor()
    yield


app = FastAPI(
    title="Hangzhou AVM API",
    description="Automated Valuation Model for Hangzhou real estate — powered by XGBoost.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: HouseInput):
    try:
        price, latency_ms = predictor.predict(body.model_dump())
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Enrichment failed — unknown district: {e}")
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Inference failed.")

    return PredictionResponse(
        predicted_price_wan=price,
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 3),
    )