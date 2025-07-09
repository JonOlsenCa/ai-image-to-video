#!/usr/bin/env python3
import warnings
import os

# Suppress the specific FutureWarnings from transformers and diffusers
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)

# Set environment variable to reduce transformers verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Now run the server
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )