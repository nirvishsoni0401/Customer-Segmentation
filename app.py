from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained KMeans model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Cluster metadata
CLUSTER_INFO = {
    0: {
        "name": "Frugal Low-Income",
        "description": "Low income, low spending. Budget-conscious shoppers — respond well to discounts and value bundles.",
        "color": "#60a5fa",
        "strategy": "Offer value deals, discount coupons, and loyalty reward points."
    },
    1: {
        "name": "Premium Spenders",
        "description": "High income, high spending. Ideal for luxury upsells, VIP programs, and early access campaigns.",
        "color": "#fbbf24",
        "strategy": "Target with premium products, exclusive memberships, and personalized offers."
    },
    2: {
        "name": "Cautious High-Earners",
        "description": "High income, low spending. Financially conservative — potential targets for premium savings products.",
        "color": "#f87171",
        "strategy": "Build trust with quality guarantees and investment-based promotions."
    },
    3: {
        "name": "Impulsive Shoppers",
        "description": "Low income, high spending. Enthusiastic buyers — credit offers and BNPL options work well.",
        "color": "#e879f9",
        "strategy": "Promote BNPL, flash sales, and limited-time offers."
    },
    4: {
        "name": "Balanced Mainstream",
        "description": "Mid income, mid spending. The average customer — broad loyalty programs suit this segment.",
        "color": "#34d399",
        "strategy": "Engage with loyalty programs, seasonal sales, and referral bonuses."
    },
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        income = float(data["income"])
        spending = float(data["spending"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input. Provide 'income' and 'spending' as numbers."}), 400

    if not (15 <= income <= 137):
        return jsonify({"error": "Annual income must be between 15 and 137 (k$)."}), 400
    if not (1 <= spending <= 99):
        return jsonify({"error": "Spending score must be between 1 and 99."}), 400

    X = np.array([[income, spending]])
    cluster_id = int(model.predict(X)[0])
    info = CLUSTER_INFO[cluster_id]

    return jsonify({
        "cluster_id": cluster_id,
        "cluster_label": cluster_id + 1,
        "name": info["name"],
        "description": info["description"],
        "color": info["color"],
        "strategy": info["strategy"],
        "input": {"income": income, "spending": spending},
    })


@app.route("/cluster-data", methods=["GET"])
def cluster_data():
    """Returns centroid coordinates and cluster metadata for the frontend charts."""
    centroids = model.cluster_centers_.tolist()
    clusters = []
    for i, centroid in enumerate(centroids):
        info = CLUSTER_INFO[i]
        clusters.append({
            "cluster_id": i,
            "cluster_label": i + 1,
            "centroid_income": round(centroid[0], 2),
            "centroid_spending": round(centroid[1], 2),
            "name": info["name"],
            "color": info["color"],
            "strategy": info["strategy"],
        })
    return jsonify({"clusters": clusters})


if __name__ == "__main__":
    app.run(debug=True)
