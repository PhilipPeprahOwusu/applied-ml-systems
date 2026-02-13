FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/list/*

# copy requirements and install python dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the application code and data
COPY src/ ./src/
COPY models/ ./models/
COPY data/customer_features_demo.csv ./data/customer_features_clustered.csv

# set environment variables
ENV PYTHONUNBUFFERED=1
ENV RECOMMENDATION_PROJECT_DIR=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]