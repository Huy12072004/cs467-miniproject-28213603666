import os
import logging
import torch
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
from model import (
    collaborative_filtering,
    content_based_filtering,
    hybrid_recommendation,
    MultiModalModel,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "shoprec-dev-secret-key")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ShopRecWeb")

# Load e-commerce datasets
users_df = pd.read_csv("users_expanded.csv")
products_df = pd.read_csv("products_expanded.csv")
product_images_df = pd.read_csv("product_images_expanded.csv")
purchases_df = pd.read_csv("purchases_expanded.csv")
browsing_df = pd.read_csv("browsing_history_expanded.csv")

# Map raw IDs safely to 0-based indices for embedding layers
unique_user_ids = sorted(users_df["user_id"].unique().tolist())
unique_product_ids = sorted(products_df["product_id"].unique().tolist())
user_to_idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
product_to_idx = {pid: idx for idx, pid in enumerate(unique_product_ids)}

# Initialize multi-modal recommendation model
mm_model = MultiModalModel(len(unique_user_ids), len(unique_product_ids))
mm_model.eval()


def _build_interacted_products(user_id: int):
    purchased_ids = purchases_df[purchases_df["user_id"] == user_id]["product_id"].unique()
    browsed_ids = browsing_df[browsing_df["user_id"] == user_id]["product_id"].unique()
    purchased_set = set(purchased_ids)

    interacted = products_df[
        products_df["product_id"].isin(purchased_ids)
        | products_df["product_id"].isin(browsed_ids)
    ].copy()
    interacted["source"] = interacted["product_id"].apply(
        lambda pid: "Purchased" if pid in purchased_set else "Browsed"
    )
    return interacted, purchased_ids, browsed_ids


def _run_multi_modal_scoring(user_id: int) -> pd.DataFrame:
    user_idx = user_to_idx.get(user_id, 0)
    mapped_product_indices = [
        product_to_idx.get(pid, 0) for pid in products_df["product_id"].tolist()
    ]

    product_tensor = torch.LongTensor(mapped_product_indices)
    user_tensor = torch.LongTensor([user_idx])
    descriptions = products_df["description"].fillna("").tolist()

    with torch.no_grad():
        embeddings = mm_model(
            user_tensor,
            product_tensor,
            descriptions,
            edge_index=None,
            product_images_df=product_images_df,
        )

    scored_df = products_df.copy()
    scored_df["score"] = embeddings.mean(dim=1).cpu().numpy()
    scored_df["source"] = "Multi-Modal"
    return scored_df.sort_values(by="score", ascending=False)


@app.route("/")
def index():
    catalog_items = products_df.to_dict(orient="records")
    return render_template("index.html", products=catalog_items)


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    try:
        user_id = int(request.form["user_id"])
        algorithm = request.form["algorithm"]
        logger.debug(f"[Recommend] user_id={user_id}, algorithm={algorithm}")

        if user_id not in users_df["user_id"].values:
            flash("User ID not found!")
            return redirect(url_for("index"))

        interacted_df, purchased_ids, browsed_ids = _build_interacted_products(user_id)
        logger.debug(f"[Recommend] Interacted product IDs: {interacted_df['product_id'].tolist()}")

        if algorithm == "collaborative":
            raw_recs = collaborative_filtering(user_id, purchases_df, products_df)
        elif algorithm == "content-based":
            raw_recs = content_based_filtering(user_id, purchases_df, browsing_df, products_df)
        elif algorithm == "hybrid":
            raw_recs = hybrid_recommendation(user_id, purchases_df, browsing_df, products_df)
        elif algorithm == "multi-modal":
            raw_recs = _run_multi_modal_scoring(user_id)
        else:
            flash("Invalid algorithm selected!")
            return redirect(url_for("index"))

        seen_ids = set(purchased_ids).union(browsed_ids)
        filtered_recs = raw_recs[~raw_recs["product_id"].isin(seen_ids)].copy()
        logger.debug(
            f"[Recommend] Filtered recommendations:\n"
            f"{filtered_recs[['product_id', 'score', 'source']]}"
        )

        if filtered_recs.empty:
            flash("No recommendations available for this user.")

        return render_template(
            "recommendations.html",
            interacted_products=interacted_df.to_dict(orient="records"),
            recommended_products=filtered_recs.to_dict(orient="records"),
        )
    except Exception as exc:
        logger.error(f"[Recommend] Error: {exc}")
        flash(f"An error occurred: {str(exc)}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)