import os
import logging
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ShopRecEngine")

VIEW_SUFFIX_MAP = {
    "_1_front": 0,
    "_2_side": 1,
    "_3_back": 2,
    "_4_full": 3,
}


def _detect_view_type(image_path: str) -> int:
    for suffix, view_idx in VIEW_SUFFIX_MAP.items():
        if suffix in image_path:
            return view_idx
    return 0


def collaborative_filtering(user_id, purchases, products, top_k=None):
    logger.debug(f"[CF] Generating recommendations for user_id={user_id}")
    user_purchases = purchases[purchases["user_id"] == user_id]["product_id"].unique()
    logger.debug(f"[CF] Purchased product_ids: {user_purchases}")

    similar_users = purchases[
        purchases["product_id"].isin(user_purchases) & (purchases["user_id"] != user_id)
    ]["user_id"].unique()
    peer_purchases = purchases[purchases["user_id"].isin(similar_users)]

    product_counts = peer_purchases["product_id"].value_counts()
    recommendations = products[
        products["product_id"].isin(product_counts.index)
        & ~products["product_id"].isin(user_purchases)
    ].copy()

    if not recommendations.empty and product_counts.max() > 0:
        recommendations["score"] = (
            recommendations["product_id"].map(product_counts).fillna(0)
            * recommendations["rating"]
        ) / product_counts.max()
    else:
        recommendations["score"] = pd.Series(dtype=float)

    recommendations["source"] = "Collaborative Filtering"
    result = recommendations.sort_values(by="score", ascending=False)
    if top_k is not None:
        result = result.head(top_k)

    logger.debug(f"[CF] Result preview:\n{result[['product_id', 'score', 'source']]}")
    return result


def content_based_filtering(user_id, purchases, browsing_history, products, top_k=None):
    logger.debug(f"[CB] Generating recommendations for user_id={user_id}")
    user_history = browsing_history[browsing_history["user_id"] == user_id]["product_id"].unique()
    logger.debug(f"[CB] Browsed product_ids: {user_history}")
    user_products = products[products["product_id"].isin(user_history)]

    if not user_products.empty and "category" in products.columns:
        recommendations = products[
            products["category"].isin(user_products["category"])
            & ~products["product_id"].isin(user_history)
        ].copy()
        avg_rating = user_products["rating"].mean()
        recommendations["score"] = (recommendations["rating"] / 5.0) * avg_rating
    else:
        recommendations = pd.DataFrame(
            columns=["product_id", "product_name", "price", "rating", "score", "source"]
        )
        logger.debug("[CB] No matching category items found.")

    recommendations["source"] = "Content-Based Filtering"
    if top_k is not None and not recommendations.empty:
        recommendations = recommendations.sort_values(by="score", ascending=False).head(top_k)

    logger.debug(f"[CB] Result preview:\n{recommendations[['product_id', 'score', 'source']]}")
    return recommendations


def hybrid_recommendation(user_id, purchases, browsing_history, products, fallback_top_n=3):
    logger.debug(f"[Hybrid] Generating recommendations for user_id={user_id}")

    user_purchases = purchases[purchases["user_id"] == user_id]["product_id"].unique()
    user_browsed = browsing_history[browsing_history["user_id"] == user_id]["product_id"].unique()
    seen_products = set(user_purchases).union(user_browsed)
    logger.debug(f"[Hybrid] Seen product_ids count={len(seen_products)}")

    collab_recs = collaborative_filtering(user_id, purchases, products)
    content_recs = content_based_filtering(user_id, purchases, browsing_history, products)

    merged_recs = pd.concat([collab_recs, content_recs], ignore_index=True)
    logger.debug(f"[Hybrid] Merged candidates:\n{merged_recs[['product_id', 'score', 'source']]}")

    if merged_recs.empty:
        logger.debug("[Hybrid] Empty candidates; falling back to top-purchased items.")
        popular_ids = purchases["product_id"].value_counts().head(fallback_top_n).index
        merged_recs = products[
            products["product_id"].isin(popular_ids) & ~products["product_id"].isin(seen_products)
        ].copy()
        merged_recs["score"] = 0.5
        merged_recs["source"] = "Popular Products"

    final_recs = (
        merged_recs.sort_values(by="score", ascending=False)
        .drop_duplicates(subset=["product_id"], keep="first")
    )
    logger.debug(f"[Hybrid] Final recommendations:\n{final_recs[['product_id', 'score', 'source']]}")
    return final_recs


class MultiModalModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim=128):
        super().__init__()
        # User & item latent embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.product_emb = nn.Embedding(num_products, embedding_dim)

        # Visual encoder (ResNet50)
        self.image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_encoder.fc = nn.Linear(2048, embedding_dim)

        # View-angle & style projection layers
        self.view_embedding = nn.Embedding(4, embedding_dim)  # front, side, back, full
        self.style_projection = nn.Linear(embedding_dim, embedding_dim)

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = ("layer4" in name) or ("fc" in name)

        # Text encoder for product descriptions
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_proj = nn.Linear(384, embedding_dim)

        # Graph layer for item relationships
        self.conv1 = GCNConv(embedding_dim, embedding_dim)

        # Multimodal fusion head
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        self._img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_image_batch(self, product_ids, product_images_df, device):
        product_id_to_info = {}
        for _, row in product_images_df.iterrows():
            path = row["image_path"]
            product_id_to_info[row["product_id"]] = {
                "path": path,
                "view_type": _detect_view_type(path),
            }

        image_tensors = []
        view_types = []
        for pid in product_ids.cpu().numpy():
            info = product_id_to_info.get(pid)
            if info and os.path.exists(info["path"]):
                try:
                    img = Image.open(info["path"]).convert("RGB")
                    image_tensors.append(self._img_transform(img))
                    view_types.append(info["view_type"])
                    logger.debug(f"[MM] Loaded image for product {pid}: {info['path']}")
                    continue
                except Exception as exc:
                    logger.error(f"[MM] Failed to load image for product {pid}: {exc}")
            else:
                logger.warning(f"[MM] Missing or invalid image for product {pid}")

            image_tensors.append(torch.zeros(3, 224, 224))
            view_types.append(0)

        image_batch = torch.stack(image_tensors).to(device)
        view_batch = torch.tensor(view_types, device=device)

        base_image_emb = self.image_encoder(image_batch)
        view_emb = self.view_embedding(view_batch)
        return self.style_projection(base_image_emb + view_emb)

    def forward(self, user_ids, product_ids, text_batch, edge_index, product_images_df=None):
        device = self.user_emb.weight.device
        user_ids = user_ids.to(device)
        product_ids = product_ids.to(device)

        user_emb = self.user_emb(user_ids)
        product_emb = self.product_emb(product_ids)

        if user_emb.dim() == 2 and product_emb.dim() == 2 and user_emb.size(0) == 1:
            user_emb = user_emb.expand(product_emb.size(0), -1)

        if product_images_df is not None:
            image_emb = self._build_image_batch(product_ids, product_images_df, device)
        else:
            image_emb = torch.zeros_like(product_emb)

        if isinstance(text_batch, list) and len(text_batch) > 0:
            text_features = self.text_encoder.encode(text_batch, convert_to_tensor=True).to(device)
            text_emb = self.text_proj(text_features)
        else:
            text_emb = torch.zeros_like(product_emb)

        if edge_index is not None:
            _ = self.conv1(product_emb, edge_index.to(device))

        cf_emb = user_emb * product_emb
        combined = torch.cat([cf_emb, image_emb, text_emb], dim=-1)
        return self.fusion(combined)
        