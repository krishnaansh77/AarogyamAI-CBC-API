from pathlib import Path
import sys
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow imports from CBC_Aarogyam/
sys.path.insert(0, str(PROJECT_ROOT))

from cbc_assessment import assess_cbc
from backend.cbc_report_parser import extract_cbc_from_report


# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------

app = FastAPI(
    title="Aarogyam AI CBC API",
    description="CBC report extraction and assessment API",
    version="1.0.0",
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://eternixx1-72ggkaihn-krishnaansh77s-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# CBC INPUT
# ------------------------------------------------------------

class CBCInput(BaseModel):
    Age: float = Field(..., ge=0, le=120)
    Height: float = Field(..., gt=0)
    Weight: float = Field(..., gt=0)
    BMI: float = Field(..., gt=0)

    Hb: float | None = None
    RBC: float | None = None
    WBC: float | None = None
    Platelets: float | None = None

    Neutrophils: float | None = None
    Lymphocytes: float | None = None
    Monocytes: float | None = None
    Eosinophils: float | None = None
    Basophils: float | None = None

    MCV: float | None = None
    MCH: float | None = None
    MCHC: float | None = None
    RDW: float | None = None


# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "Aarogyam AI CBC API",
        "status": "running",
        "docs": "/docs",
    }


# ------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "cbc-assessment",
    }


# ------------------------------------------------------------
# PARSE CBC REPORT
# ------------------------------------------------------------

@app.post("/api/cbc/parse-report")
async def parse_cbc_report(
    file: UploadFile = File(...)
):
    allowed_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
    }

    filename = file.filename or "uploaded_report"
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, PNG, JPG and JPEG files "
                "are supported."
            ),
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

        result = extract_cbc_from_report(
            temp_path
        )

        result["source_file"] = filename

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"CBC report extraction failed: {error}",
        )

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# ------------------------------------------------------------
# CBC ASSESSMENT
# ------------------------------------------------------------

@app.post("/api/cbc/assess")
def assess_cbc_endpoint(
    data: CBCInput
):
    try:
        patient_data = data.model_dump()

        result = assess_cbc(
            patient_data
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"CBC assessment failed: {error}",
        )