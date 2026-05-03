"""
FastAPI application for the Churn Prevention System.
Serves churn predictions and personalized product recommendations.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse, ProductRecommendation
from api.inference import InferenceEngine

# Initialize FastAPI
app = FastAPI(
    title="Churn Prevention System API",
    description="Predict customer churn and generate personalized product recommendations for retention.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load inference engine at startup
engine = None


@app.on_event("startup")
async def startup():
    global engine
    engine = InferenceEngine()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and loaded models."""
    return HealthResponse(
        status="healthy",
        models_loaded=engine.loaded if engine else {},
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict churn probability and generate product recommendations.

    Pipeline: Churn Prediction → FAISS Retrieval → NeuMF Ranking → (Optional) LLM Reranking
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        result = engine.predict(
            customer_id=request.customer_id,
            top_k=request.top_k,
            use_llm=request.use_llm_reranker,
        )

        return PredictionResponse(
            customer_id=result["customer_id"],
            churn_probability=result["churn_probability"],
            churn_risk_level=result["churn_risk_level"],
            p_alive=result.get("p_alive"),
            predicted_clv=result.get("predicted_clv"),
            recommendations=[
                ProductRecommendation(**rec) for rec in result["recommendations"]
            ],
            retention_action=result["retention_action"],
            model_info=result["model_info"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers")
async def list_customers():
    """List available customer IDs (sample)."""
    if engine is None or not engine.loaded["retrieval"]:
        raise HTTPException(status_code=503, detail="Models not loaded")

    user_ids = list(engine.retrieval_mappings["user_to_idx"].keys())[:100]
    return {"customer_ids": user_ids, "total": len(engine.retrieval_mappings["user_to_idx"])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
