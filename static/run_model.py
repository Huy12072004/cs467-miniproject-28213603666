import logging
import pandas as pd
import torch
from model import MultiModalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShopRecRunner")


def main():
    # Load e-commerce data files
    users_df = pd.read_csv("users_expanded.csv")
    products_df = pd.read_csv("products_expanded.csv")
    product_images_df = pd.read_csv("product_images_expanded.csv")

    # Map IDs safely to 0-based indices for embedding layers
    unique_user_ids = sorted(users_df["user_id"].unique().tolist())
    unique_product_ids = sorted(products_df["product_id"].unique().tolist())
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    product_to_idx = {pid: idx for idx, pid in enumerate(unique_product_ids)}

    # Initialize multi-modal model
    num_users = len(unique_user_ids)
    num_products = len(unique_product_ids)
    mm_model = MultiModalModel(num_users, num_products)
    mm_model.eval()

    # Prepare sample inference inputs
    target_user_id = 1
    user_idx = user_to_idx.get(target_user_id, 0)
    mapped_product_indices = [
        product_to_idx.get(pid, 0) for pid in products_df["product_id"].tolist()
    ]

    user_tensor = torch.LongTensor([user_idx])
    product_tensor = torch.LongTensor(mapped_product_indices)
    descriptions = products_df["description"].fillna("").tolist()

    # Run model inference
    with torch.no_grad():
        outputs = mm_model(
            user_tensor,
            product_tensor,
            descriptions,
            edge_index=None,
            product_images_df=product_images_df,
        )

    # Print shape diagnostics
    logger.info("Inference completed successfully.")
    print("Outputs shape:", outputs.shape)
    print("Mean dim=0:", outputs.mean(dim=0).shape)
    print("Mean dim=1:", outputs.mean(dim=1).shape)
    print("Products shape:", products_df.shape)


if __name__ == "__main__":
    main()