# Senior ML Engineer Interview Prep: Two-Stage Offer Recommender

## 1. Project Overview (The STAR Method)
**Situation:** At GoAuto, the marketing team needed to move beyond guesswork to match 8M+ customers with relevant service offers (oil changes, loyalty rewards). We had a critical production bug: the ranking model was outputting identical scores for different offers, breaking personalization.
**Task:** Design, build, and deploy an end-to-end Offer Recommendation System that is accurate, fast (<100ms), and fully automated.
**Action:** 
- **Architecture:** Implemented a two-stage Retrieval (K-Means) and Ranking (LightGBM) pipeline.
- **Feature Engineering:** Resolved the "identical scores" bug by engineering **Interaction Features** (e.g., `Frequency x Offer_Type`).
- **MLOps:** Automated the lifecycle using **Airflow** for orchestration and **MLflow** for experiment tracking/model registry.
- **Deployment:** Containerized the service with **Docker** and deployed to **AWS App Runner** with **Redis** for low-latency feature lookups.
**Result:** Achieved a **0.91 AUC** and maintained **P99 latency <100ms**, generating a projected **15% lift** in conversion rates.

---

## 2. Technical Deep-Dive Questions

### Q: How is the recommendation score calculated?
**Answer:** The score is a **predicted probability of redemption** (0 to 1). It is calculated by a **LightGBM Ranking Model** that evaluates three feature types:
1. **Customer RFM:** Recency, Frequency, and Monetary value.
2. **Offer Attributes:** Type (Discount/Loyalty) and Value ($).
3. **Interaction Features:** The "Secret Sauce." We multiply user behavior by offer type (e.g., `Frequency * Is_Discount`). This allows the model to learn non-linear affinities, such as "High-frequency users prefer Loyalty points over one-off discounts."

### Q: How do you explain the output for Cluster 2 (High Score: 0.86)?
**Answer:** 
- **The Segment:** Cluster 2 represents our **"Power Users"**—loyal customers with high visit frequency.
- **The Strategy:** The Retrieval stage identifies them as loyalists. The Ranking stage then prioritizes **Loyalty Points (Score: 0.86)** over deep discounts. 
- **Business Impact:** This optimizes ROI by rewarding loyalty with points (low cost) rather than giving away high-margin services (oil changes) to someone who was likely to visit anyway.

### Q: How do you explain the output for Cluster 4 (Low Score: 0.02)?
**Answer:** 
- **The Context:** In automotive marketing, a 2.6% conversion is actually strong (industry standard is 0.5% - 2%).
- **Relative Ranking:** Even with low absolute probabilities, the **Relative Ranking** is what matters. The model is 10% more confident in a "$25 Off" coupon for this user than a "Winter Package," allowing us to prioritize the most efficient nudge.

### Q: Why did you use two stages (Retrieval & Ranking)?
**Answer:** It is a balance of **Scale vs. Precision**. 
- **Retrieval:** Uses K-Means to filter 100+ possible offers down to the top 5–10 candidates in <5ms.
- **Ranking:** Uses a complex LightGBM model to perform "heavy lifting" on only those 5–10 candidates. This architecture is what allows us to serve 8M+ customers with **sub-100ms latency**.

---

## 3. MLOps & Infrastructure

### Q: What Airflow DAGs did you implement?
**Answer:** I built a unified `recommender_pipeline` DAG with four key tasks:
1. **Generate Data:** Daily ETL of transaction logs.
2. **Feature Engineering:** Calculation of RFM and Interaction features.
3. **Train Model:** Retrains the LightGBM model and logs metrics/artifacts to **MLflow**.
4. **Load to Redis:** Pushes the fresh features to the production cache.
*Design Note:* I chained these to ensure **Data Consistency**—we never update the cache unless the model training succeeds.

### Q: How do you monitor and handle Model Drift?
**Answer:** I use a closed-loop monitoring process:
- **Metric:** I calculate the **Population Stability Index (PSI)** for key interaction features.
- **Threshold:** If **PSI > 0.2**, it signals a significant shift in customer behavior.
- **Automation:** The Airflow monitor triggers a Slack alert and initiates an automated retraining job to align the model with the new data distribution.

### Q: How did you deploy the system to production?
**Answer:**
1. **Docker:** Containerized the FastAPI service for environment parity.
2. **AWS App Runner:** Used for serverless, auto-scaling execution of the API.
3. **Amazon ElastiCache (Valkey):** Used as our high-speed feature store.
4. **GitHub Actions:** Built a CI/CD pipeline (`deploy.yml`) that builds, tests, and deploys the image to AWS ECR on every push to `main`.

---

## 4. Evaluation & Metrics

### Q: What metrics did you use and why?
- **AUC-ROC (0.91):** Measures ranking quality. High AUC ensures the best offers are always at the top.
- **Recall@K:** Ensures the Retrieval stage doesn't "miss" relevant offers.
- **P99 Latency (<100ms):** Ensures a high-quality, instantaneous user experience.
- **Precision at Threshold:** Protects the brand by only sending offers we are highly confident in (anti-spam).
- **Conversion Lift (+15%):** The ultimate business KPI—incremental revenue generated by moving from "Most Popular" to "Personalized" offers.
