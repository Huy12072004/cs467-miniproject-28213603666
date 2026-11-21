import pandas as pd
import torch
from model import (
    collaborative_filtering,
    content_based_filtering,
    hybrid_recommendation,
    MultiModalModel,
)


def test_recommender_pipeline():
    # Load e-commerce datasets
    users_df = pd.read_csv("users_expanded.csv")
    products_df = pd.read_csv("products_expanded.csv")
    product_images_df = pd.read_csv("product_images_expanded.csv")
    purchases_df = pd.read_csv("purchases_expanded.csv")
    browsing_df = pd.read_csv("browsing_history_expanded.csv")

    # Safe 0-based index mapping for embedding layers
    unique_user_ids = sorted(users_df["user_id"].unique().tolist())
    unique_product_ids = sorted(products_df["product_id"].unique().tolist())
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    product_to_idx = {pid: idx for idx, pid in enumerate(unique_product_ids)}

    sample_user_id = unique_user_ids[0] if unique_user_ids else 1

    # Verify traditional & hybrid recommendation functions
    cf_recs = collaborative_filtering(sample_user_id, purchases_df, products_df)
    cb_recs = content_based_filtering(sample_user_id, purchases_df, browsing_df, products_df)
    hyb_recs = hybrid_recommendation(sample_user_id, purchases_df, browsing_df, products_df)

    assert isinstance(cf_recs, pd.DataFrame)
    assert isinstance(cb_recs, pd.DataFrame)
    assert isinstance(hyb_recs, pd.DataFrame)

    # Initialize and run MultiModalModel inference
    num_users = len(unique_user_ids)
    num_products = len(unique_product_ids)
    mm_model = MultiModalModel(num_users, num_products)
    mm_model.eval()

    user_idx = user_to_idx.get(sample_user_id, 0)
    mapped_product_indices = [
        product_to_idx.get(pid, 0) for pid in products_df["product_id"].tolist()
    ]

    user_tensor = torch.LongTensor([user_idx])
    product_tensor = torch.LongTensor(mapped_product_indices)
    descriptions = products_df["description"].fillna("").tolist()

    with torch.no_grad():
        outputs = mm_model(
            user_tensor,
            product_tensor,
            descriptions,
            edge_index=None,
            product_images_df=product_images_df,
        )

    assert outputs.shape[0] == len(products_df)

    # Print shape diagnostics (kept consistent with original output)
    print("Outputs shape:", outputs.shape)
    print("Mean dim=0:", outputs.mean(dim=0).shape)
    print("Mean dim=1:", outputs.mean(dim=1).shape)
    print("Products shape:", products_df.shape)
    print(
        f"Counts -> CF: {len(cf_recs)}, CB: {len(cb_recs)}, Hybrid: {len(hyb_recs)}"
    )


if __name__ == "__main__":
    test_recommender_pipeline()