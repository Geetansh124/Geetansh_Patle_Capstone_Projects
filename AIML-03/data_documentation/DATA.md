# AIML-03: Dataset Documentation & Split Methodology

**Project Goal:** Large-Scale Personalized Recommendation Engine  
**Dataset:** MovieLens 25M Benchmark (GroupLens Research)

---

## 1. Dataset Overview & Provenance

- **Source:** GroupLens Research, University of Minnesota
- **Access URL:** [GroupLens MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)
- **Total Volume:** 25,000,095 ratings and 1,093,360 tag applications across 62,423 movies by 162,541 users.
- **Timestamp Coverage:** January 09, 1995 to November 21, 2019.
- **Target Path:** `data/ml-25m/ratings.csv` and `data/ml-25m/movies.csv`

---

## 2. Relational Schemas

### `ratings.csv`
| Column | Type | Range / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `userId` | int64 | 1 to 162,541 | Unique identifier for user |
| `movieId` | int64 | 1 to 209,171 | Unique identifier for movie |
| `rating` | float32 | 0.5 to 5.0 (0.5 steps) | Star rating assigned by user |
| `timestamp` | int64 | Unix epoch seconds | Monotonic sequence timestamp |

### `movies.csv`
| Column | Type | Format | Description |
| :--- | :--- | :--- | :--- |
| `movieId` | int64 | Integer | Primary key |
| `title` | string | `Title (Release Year)` | Film title and metadata |
| `genres` | string | Pipe-delimited | 19 categorical genres |

---

## 3. Chronological Split Methodology & Leakage Prevention

In sequential recommender systems, **random cross-validation causes fatal future data leakage**. Scoring past items using future interaction embeddings distorts realistic online performance.

### Partitioning Protocol
1. **Within-User Temporal Ordering:** Interactions for every user are sorted strictly by `timestamp` ascending:
   $$\mathcal{I}_u = (i_1, t_1), (i_2, t_2), \dots, (i_N, t_N) \quad \text{where } t_1 \le t_2 \le \dots \le t_N$$
2. **Train/Test Cutoff:**
   - The first 80% of chronological events are assigned to training $\mathcal{I}_u^{\text{train}}$.
   - The latest 20% of chronological events are held out for testing $\mathcal{I}_u^{\text{test}}$.
   - Cutoff boundary: $K_{\text{cutoff}} = \max(1, \lfloor 0.8 \times N \rfloor)$.
3. **Cold-Start Policy:**
   - Single-interaction users ($N=1$) remain in the training partition because within-user holdout evaluation is mathematically impossible.
   - Users with sparse interactions ($2 \le N < 5$) retain exactly one training event and one test event.
4. **Target Candidate Pool:**
   - In ranking evaluation, models must rank unseen candidate items from the global catalog: $\mathcal{C}_u = \mathcal{C} \setminus \mathcal{I}_u^{\text{train}}$.
   - Already-consumed items are filtered from recommendations.

---

## 4. Evaluation Metrics Formulation

Evaluation is conducted on the held-out test interactions $\mathcal{I}_u^{\text{test}}$ for each user:

1. **Recall@K:**
   $$\text{Recall@K}(u) = \frac{|\text{TopK}(u) \cap \mathcal{I}_u^{\text{test}}|}{|\mathcal{I}_u^{\text{test}}|}$$

2. **Normalized Discounted Cumulative Gain (NDCG@K):**
   $$\text{DCG@K}(u) = \sum_{r=1}^{K} \frac{\mathbb{I}(\text{TopK}_r(u) \in \mathcal{I}_u^{\text{test}})}{\log_2(r + 1)}$$
   $$\text{IDCG@K}(u) = \sum_{r=1}^{\min(K, |\mathcal{I}_u^{\text{test}}|)} \frac{1}{\log_2(r + 1)}$$
   $$\text{NDCG@K}(u) = \frac{\text{DCG@K}(u)}{\text{IDCG@K}(u)}$$

---

## 5. Offline Benchmark vs. Online CTR Limitations

- **No Online CTR:** Offline Recall@K and NDCG@K measure historical ranking recovery. They **do not prove online Click-Through Rate (CTR)**, conversion rate, or real user satisfaction.
- **Popularity Bias:** Popularity-based models achieve misleadingly high metrics on datasets where user engagement is heavily concentrated on blockbusters.
- **Cold-Start Drift:** Production traffic encounters novel users and items without interaction histories, requiring fallback logic.
