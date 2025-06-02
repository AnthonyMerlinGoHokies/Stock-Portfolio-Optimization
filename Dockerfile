# BUILD STAGE - Install all dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
COPY conda.yaml .

# Install dependencies to user directory
RUN pip install --user --no-cache-dir -r requirements.txt

# PRODUCTION STAGE - Final lightweight image
FROM python:3.11-slim

# Install curl for health checks (as root)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r optimizer && useradd -r -g optimizer optimizer

# Create matplotlib and MLflow directories and set permissions
RUN mkdir -p /home/optimizer/.config/matplotlib \
             /home/optimizer/.cache/matplotlib \
             /home/optimizer/.mlflow \
             /home/optimizer/app/mlruns && \
    chown -R optimizer:optimizer /home/optimizer

# Set working directory
WORKDIR /home/optimizer/app

# Copy only the installed packages from builder stage
COPY --from=builder /root/.local /home/optimizer/.local

# Copy application files and set ownership
COPY --chown=optimizer:optimizer enhanced_portfolio_optimizer.py .
COPY --chown=optimizer:optimizer mlflow_project_optimizer.py .
COPY --chown=optimizer:optimizer model_registry_manager.py .
COPY --chown=optimizer:optimizer model_registry_demo.py .
COPY --chown=optimizer:optimizer MLproject .
COPY --chown=optimizer:optimizer conda.yaml .

# Create necessary directories
RUN mkdir -p /home/optimizer/app/data \
             /home/optimizer/app/output \
             /home/optimizer/app/mlruns \
             /home/optimizer/app/artifacts && \
    chown -R optimizer:optimizer /home/optimizer/app

# Switch to non-root user
USER optimizer

# Set environment variables
ENV PATH=/home/optimizer/.local/bin:$PATH
ENV PYTHONPATH=/home/optimizer/app
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# MLflow-specific environment variables
ENV MLFLOW_TRACKING_URI=file:///home/optimizer/app/output/mlruns
ENV MLFLOW_REGISTRY_URI=file:///home/optimizer/app/output/mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/home/optimizer/app/output/artifacts

# Portfolio-specific environment variables with defaults
ENV DATA_PATH=/home/optimizer/app/archive
ENV OUTPUT_PATH=/home/optimizer/app/output
ENV MODEL_TYPE=ridge
ENV TOP_STOCKS=10
ENV RANDOM_SEED=42
ENV LOG_LEVEL=INFO

# Expose MLflow UI port
EXPOSE 5000

# Add health check that includes MLflow
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import pandas, numpy, sklearn, mlflow; print('Health check passed')" || exit 1

# Default command runs the enhanced optimizer
CMD ["python", "enhanced_portfolio_optimizer.py"]