"""
FastAPI application for the Churn Prevention System.
Serves churn predictions and personalized product recommendations.

Production hardening: lifespan startup, API-key auth, Prometheus metrics,
structured logging, and request IDs.
"""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.inference import InferenceEngine
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse, ProductRecommendation
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("cjp.api")

# ---- Prometheus metrics ----
PREDICTION_REQUESTS = Counter("cjp_prediction_requests_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("cjp_prediction_errors_total", "Total prediction errors")
PREDICTION_LATENCY = Histogram("cjp_prediction_latency_seconds", "Prediction latency (s)")
CHURN_RISK = Counter("cjp_churn_risk_total", "Predictions by churn risk level", ["risk_level"])

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Loading inference engine...")
    engine = InferenceEngine()
    logger.info("Inference engine ready: %s", engine.loaded)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Churn Prevention System API",
    description="Predict customer churn and generate personalized product recommendations.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["x-request-id"] = request_id
    logger.info(
        "request id=%s method=%s path=%s status=%s ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, elapsed * 1000,
    )
    return response


def require_api_key(x_api_key: str = Header(default="")):
    """Simple API-key auth for write/predict endpoints."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and loaded models. (No auth.)"""
    return HealthResponse(
        status="healthy",
        models_loaded=engine.loaded if engine else {},
        version="2.0.0",
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
async def predict(request: PredictionRequest):
    """
    Predict churn probability and generate product recommendations.

    Pipeline: Churn Prediction -> FAISS Retrieval -> NeuMF Ranking -> (Optional) LLM Reranking
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    PREDICTION_REQUESTS.inc()
    with PREDICTION_LATENCY.time():
        try:
            result = engine.predict(
                customer_id=request.customer_id,
                top_k=request.top_k,
                use_llm=request.use_llm_reranker,
            )
        except Exception as e:
            PREDICTION_ERRORS.inc()
            logger.exception("prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    CHURN_RISK.labels(risk_level=result["churn_risk_level"]).inc()

    return PredictionResponse(
        customer_id=result["customer_id"],
        churn_probability=result["churn_probability"],
        churn_risk_level=result["churn_risk_level"],
        p_alive=result.get("p_alive"),
        predicted_clv=result.get("predicted_clv"),
        recommendations=[ProductRecommendation(**rec) for rec in result["recommendations"]],
        retention_action=result["retention_action"],
        model_info=result["model_info"],
    )


@app.get("/customers", dependencies=[Depends(require_api_key)])
async def list_customers():
    """List available customer IDs (sample)."""
    if engine is None or not engine.loaded["retrieval"]:
        raise HTTPException(status_code=503, detail="Models not loaded")

    user_ids = list(engine.retrieval_mappings["user_to_idx"].keys())[:100]
    return {"customer_ids": user_ids, "total": len(engine.retrieval_mappings["user_to_idx"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
