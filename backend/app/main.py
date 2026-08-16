import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import trial_routes, patient_routes, matching_routes, analysis_routes, research_routes, ocr_routes

app = FastAPI(
    title="AI-Powered Clinical Research Assistant API",
    description="Backend services for patient-trial matching and clinical outcome prediction",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon prototype; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(trial_routes.router, prefix="/api/trials", tags=["Trials"])
app.include_router(patient_routes.router, prefix="/api/patients", tags=["Patients"])
app.include_router(matching_routes.router, prefix="/api/matching", tags=["Matching"])
app.include_router(analysis_routes.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(research_routes.router, prefix="/api/research", tags=["Research"])
app.include_router(ocr_routes.router, prefix="/api/ocr", tags=["OCR"])

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Clinical Research Assistant API",
        "status": "online",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
