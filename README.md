# Two-Stage Offer Recommendation System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Architecture](https://img.shields.io/badge/Architecture-Two--Stage%20(Retrieval%20%2B%20Ranking)-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-blue)
![AWS](https://img.shields.io/badge/Cloud-AWS%20App%20Runner-orange)
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-purple)

A production-grade **Two-Stage Recommender System** designed to personalize service offers for **8M+ automotive customers** in real-time. The system employs a **Retrieval layer** (Candidate Generation via Clustering) to narrow down the search space, followed by a **Ranking layer** (LightGBM) to score probabilities with high precision.

**[Live API Documentation](https://prsmyk6rjm.us-east-1.awsapprunner.com/docs)**

---

## 🏗️ System Architecture: The Two-Stage Approach

To handle scale efficiently while maintaining high accuracy, the system splits the recommendation problem into two distinct stages:

### **Stage 1: Retrieval (Candidate Generation)**
*   **Goal:** Quickly filter the universe of all possible offers down to a relevant subset (Top-K candidates).
*   **Current Method:** **K-Means Clustering** groups customers into behavioral segments based on RFM (Recency, Frequency, Monetary) and service history.
*   **Outcome:** Reduces the scoring load on the heavy ranking model by only passing relevant candidates.

### **Stage 2: Fine Ranking**
*   **Goal:** Accurately predict the probability of redemption for each candidate.
*   **Method:** A **LightGBM Gradient Boosting** model scores the candidates using complex **Interaction Features** (e.g., `frequency_x_offer_type`, `recency_x_offer_value`).
*   **Performance:** Achieves **0.83 AUC-ROC** on the hold-out test set.

---

## 🚀 Key Features

*   **Real-Time Serving:** Deployed as a **FastAPI** microservice with **Redis** caching for feature lookup, achieving sub-100ms latency.
*   **Automated MLOps:** Full data lifecycle orchestration (ETL -> Feature Engineering -> Training) using **Apache Airflow**.
*   **Experiment Tracking:** Integrated **MLflow** to track model versions, metrics, and artifacts automatically.
*   **Advanced Data Engineering:** Implemented **Graph-based Entity Resolution** (using NetworkX) to unify 950k raw records into ~860k unique customer profiles.
*   **Cloud-Native:** Containerized with **Docker** and deployed to **AWS App Runner** via a custom **GitHub Actions CI/CD** pipeline.

---

## 🛠️ Tech Stack

*   **Architecture:** Two-Stage Recommender (Retrieval & Ranking)
*   **Machine Learning:** LightGBM, Scikit-learn (K-Means), Pandas, NumPy
*   **API Framework:** FastAPI, Uvicorn
*   **Data Engineering:** NetworkX (Graph Theory), Apache Airflow
*   **MLOps:** MLflow, Docker, GitHub Actions
*   **Infrastructure:** AWS App Runner, AWS ECR, Redis

---

## 📊 Results

*   **Ranking Accuracy:** 0.83 AUC-ROC.
*   **Business Impact:** Simulated 10x lift in offer conversion rate (from ~3% baseline to ~30%) by targeting high-propensity users.
*   **Latency:** <100ms p95 latency for real-time inference.

---

## 💻 Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/offer-recommendation-system.git
    cd offer-recommendation-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```

3.  **Run the API locally:**
    ```bash
    uvicorn src.main:app --reload
    ```
    Access the interactive docs at `http://localhost:8000/docs`.

4.  **Run with Docker:**
    ```bash
    docker build -t offer-recommender .
    docker run -p 8000:8000 offer-recommender
    ```

---

## 📈 Future Roadmap: FAISS Retrieval

We are currently planning to upgrade the **Retrieval Stage** (Stage 1) to leverage **Vector Search**:

*   **Objective:** Implement **User-to-User Collaborative Filtering** using **FAISS (Facebook AI Similarity Search)**.
*   **Method:** Instead of clustering users into fixed segments, we will compute dense embeddings for each user based on their feature vector.
*   **Outcome:** At inference time, we will query the FAISS index to find the exact K-Nearest Neighbors for the target user and recommend items popular within that specific local neighborhood. This will provide significantly more granular personalization than K-Means.

---

**Author:** [Your Name]
**License:** MIT