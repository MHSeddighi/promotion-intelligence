from fastapi import FastAPI
from app.api.campaigns import router as campaigns_router
from app.api.analytics import router as analytics_router
from app.api.recommendations import router as recommendations_router

app = FastAPI(
    title="Promotion Intelligence API",
    description="AI-powered promotion analytics and campaign recommendation system",
    version="0.1.0",
)

app.include_router(campaigns_router)
app.include_router(analytics_router)
app.include_router(recommendations_router)


@app.get("/")
def root():
    return {
        "service": "Promotion Intelligence Backend",
        "version": "0.1.0",
        "endpoints": {
            "campaigns": "/campaigns",
            "campaign_detail": "/campaigns/{id}",
            "campaign_impact": "/analytics/campaigns/{id}/impact",
            "campaign_cannibalization": "/analytics/campaigns/{id}/cannibalization",
            "campaign_forecast": "/analytics/campaigns/{id}/forecast",
            "recommendations": "/recommendations",
            "best_campaigns": "/recommendations/best",
            "worst_campaigns": "/recommendations/worst",
            "patterns": "/recommendations/patterns",
            "scenario": "/recommendations/scenario",
        },
    }
