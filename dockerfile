FROM nvcr.io/nvidia/pytorch:24.01-py3-arm64

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (faster than pip)
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

