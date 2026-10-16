

\# Local Qwen FastAPI



A local AI text-generation REST API built with FastAPI, PyTorch, Hugging Face Transformers, and Qwen2.5-0.5B-Instruct.



\## Features



\- Local Qwen model inference

\- FastAPI REST endpoints

\- Swagger UI documentation

\- API-key authentication

\- Health-check endpoint

\- Configurable response length



\## Technology Stack



\- Python

\- FastAPI

\- Uvicorn

\- PyTorch

\- Hugging Face Transformers

\- Qwen2.5-0.5B-Instruct



\## Setup



\### 1. Create and activate a virtual environment



```cmd

py -3.13 -m venv .venv

.venv\\Scripts\\activate

```



\### 2. Install dependencies



```cmd

pip install -r requirements.txt

```



\### 3. Configure the API key



In Windows CMD:



```cmd

set AI\_API\_KEY=replace-with-a-strong-random-key

```



\### 4. Start the server



```cmd

python main.py

```



\### 5. Open API documentation



http://127.0.0.1:8000/docs



\## Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | /health | Check API and model status |

| POST | /generate | Generate text with Qwen |



\## Security



Never commit API keys, virtual environments, or downloaded model files to GitHub. This project is configured for local development and is not publicly deployed.



