# SegmentIQ — Customer Segmentation Dashboard

A K-Means clustering web app that segments mall customers by **Annual Income** and **Spending Score** into 5 distinct personas — with an interactive prediction tool powered by a Flask backend.

---

## Project Structure

```
customer-segmentation-app/
│
├── app.py                          # Flask backend + /predict API
├── model.pkl                       # Trained KMeans model (k=5)
├── Customer_Segmentation.ipynb     # Original analysis notebook
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Frontend dashboard
└── static/
    └── style.css                   # Stylesheet
```

---

## Cluster Profiles

| Cluster | Name                  | Income | Spending |
|---------|-----------------------|--------|----------|
| 1       | Frugal Low-Income     | Low    | Low      |
| 2       | Premium Spenders      | High   | High     |
| 3       | Cautious High-Earners | High   | Low      |
| 4       | Impulsive Shoppers    | Low    | High     |
| 5       | Balanced Mainstream   | Mid    | Mid      |

**Model Metrics:**
- Silhouette Score: `0.554`
- Davies-Bouldin Score: `0.443`
- WCSS (k=5): `44,448`

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation-app.git
cd customer-segmentation-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## Regenerate model.pkl (optional)

If you want to retrain on your own `Mall_Customers.csv`:

```python
import pandas as pd
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

with open('model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("Model saved.")
```

---

## Deploy to Render (Free Tier)

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set these options:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3

That's it — Render auto-deploys on every push to `main`.

---

## Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
```

---

## API Reference

### `POST /predict`

Classifies a customer into a segment.

**Request body:**
```json
{ "income": 70, "spending": 82 }
```

**Response:**
```json
{
  "cluster_id": 1,
  "cluster_label": 2,
  "name": "Premium Spenders",
  "description": "High income, high spending. Ideal for luxury upsells...",
  "color": "#fbbf24",
  "strategy": "Target with premium products and exclusive memberships.",
  "input": { "income": 70, "spending": 82 }
}
```

### `GET /cluster-data`

Returns all cluster centroids and metadata.

---

## Tech Stack

- **Backend:** Python, Flask, scikit-learn
- **Frontend:** HTML, CSS, Chart.js
- **Model:** K-Means Clustering (k=5, k-means++ init)
- **Dataset:** Mall Customers (200 records)
