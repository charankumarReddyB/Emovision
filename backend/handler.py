# backend/handler.py
"""Netlify Function entrypoint for the FastAPI backend.
Netlify expects an ASGI callable named `handler`. We simply re‑export the
FastAPI `app` defined in `backend/main.py`.
"""

# Import the FastAPI app that is already fully configured
from main import app as handler

# Optional: allow local testing with `python -m uvicorn handler:handler`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(handler, host="0.0.0.0", port=8000)
