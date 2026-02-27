# Two-Stage Offer Recommendation System

A production-grade recommendation engine designed to personalize service offers for 8M+ automotive customers. The system utilizes a two-stage architecture: a retrieval layer for candidate generation and a gradient-boosted ranking layer for precision scoring.

## System Architecture

The architecture decouples retrieval from ranking to maintain sub-100ms latency at scale.

### Stage 1: Retrieval (Candidate Generation)
*   **Objective:** Reduce the search space by identifying a relevant subset of offers for each user.
*   **Methodology:** K-Means clustering segments customers based on RFM (Recency, Frequency, Monetary) and service history.
*   **Outcome:** Filters the universe of all possible offers down to the Top-K most probable candidates for each segment.

### Stage 2: Fine Ranking
*   **Objective:** Predict the exact probability of redemption for the retrieved candidates.
*   **Methodology:** A LightGBM model scores candidates using interaction features (e.g., `frequency_x_offer_type`) to capture user-offer affinity.
*   **Performance:** Achieves 0.83 AUC-ROC on hold-out validation sets.

## Key Features

*   **Real-Time Serving:** FastAPI microservice with Redis-backed feature caching for low-latency inference.
*   **Automated MLOps:** End-to-end orchestration (ETL, Feature Engineering, Training) via Apache Airflow.
*   **Experiment Tracking:** MLflow integration for model versioning, metric tracking, and artifact management.
*   **Entity Resolution:** Graph-based customer deduplication using NetworkX, unifying 950k raw records into unique profiles.
*   **Cloud Infrastructure:** Containerized with Docker and deployed to AWS App Runner via GitHub Actions.

## Tech Stack

*   **Modeling:** LightGBM, Scikit-learn, Pandas, NumPy
*   **API Framework:** FastAPI, Uvicorn
*   **Orchestration:** Apache Airflow, MLflow
*   **Infrastructure:** Docker, Redis, AWS App Runner

## Results

*   **Model Metric:** 0.83 AUC-ROC.
*   **Projected Impact:** 10x lift in offer conversion rate based on offline simulation.
*   **Performance:** <100ms p95 latency for real-time inference.

## Live API & Usage

The service is deployed on AWS App Runner and can be queried via the following endpoint:

**Base URL:** `https://prsmyk6rjm.us-east-1.awsapprunner.com`

### Quick Start (API Examples)

To get personalized recommendations for a specific customer, use the `/recommend/{customer_id}` endpoint:

1.  **High-Confidence Recommendation (Loyalty):**
    ```bash
    curl -X 'GET' 'https://prsmyk6rjm.us-east-1.awsapprunner.com/recommend/0052PXPOJC'
    ```
    *Returns high-scoring loyalty point offers for a "Power User."*

2.  **High-Confidence Recommendation (Discount):**
    ```bash
    curl -X 'GET' 'https://prsmyk6rjm.us-east-1.awsapprunner.com/recommend/0023HAQRAD'
    ```
    *Returns strong maintenance discounts for a price-sensitive visitor.*

3.  **Personalized Ranking Test:**
    ```bash
    curl -X 'GET' 'https://prsmyk6rjm.us-east-1.awsapprunner.com/recommend/0003QF4K5R'
    ```
    *Returns a diverse set of ranked offers based on complex behavior interactions.*

## Setup and Deployment

1.  **Environment Configuration:**
    ```bash
    conda create -n offer-recommender python=3.9
    conda activate offer-recommender
    pip install -r requirements.txt
    ```

2.  **Infrastructure:**
    ```bash
    docker compose up -d redis
    ```

3.  **Local Execution:**
    ```bash
    uvicorn src.main:app --reload
    ```

## Future Roadmap

*   **Vector Retrieval:** Transitioning Stage 1 to a Vector Search architecture using FAISS or Milvus.
*   **Deep Ranking:** Evaluating Neural Collaborative Filtering (NCF) for the ranking stage to capture non-linear feature interactions.

**Author:** Philip Owusu
**License:** MIT
