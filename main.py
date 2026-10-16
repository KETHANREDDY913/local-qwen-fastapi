
import os
import secrets
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Configure Hugging Face cache BEFORE importing Transformers
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "hf_cache"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_XET_CACHE"] = str(CACHE_DIR / "xet")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = None
model = None


# --------------------------------------------------
# Load the model when the API starts
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    try:
        logger.info("Loading tokenizer from %s", MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=str(CACHE_DIR),
        )

        logger.info("Loading Qwen model...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(CACHE_DIR),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        model.eval()
        logger.info("Qwen model loaded successfully.")

        yield

    except Exception:
        logger.exception("Failed to load Qwen model.")
        raise

    finally:
        model = None
        tokenizer = None
        logger.info("Model resources released.")


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Local Qwen AI API",
    description=(
        "A local text-generation API powered by "
        "Qwen2.5-0.5B-Instruct."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=200, ge=1, le=512)


class GenerateResponse(BaseModel):
    answer: str
    model: str


# --------------------------------------------------
# API-key authentication
# --------------------------------------------------

def verify_api_key(x_api_key: str | None) -> None:
    expected_key = os.environ.get("AI_API_KEY")

    if not expected_key:
        logger.error("AI_API_KEY is not configured.")

        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
        )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# Text-generation endpoint
# --------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
):
    verify_api_key(x_api_key)

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot contain only whitespace.",
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer clearly and accurately."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[
            0
        ][inputs["input_ids"].shape[1]:]

        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return GenerateResponse(
            answer=answer,
            model=MODEL_ID,
        )

    except Exception:
        logger.exception("Text generation failed.")

        raise HTTPException(
            status_code=500,
            detail="Text generation failed. Check server logs.",
        )


# --------------------------------------------------
# Run the API locally
# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
