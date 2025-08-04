import pandas as pd
import numpy as np
from clip_model.clip_extractor import extract_clip_features
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import os
import joblib

def save_similarity_db(df, X, file_path="similar_ads_db.npz"):
    df = df.reset_index(drop=True)
    df["embedding"] = list(X)
    np.savez_compressed(file_path, metadata=df.to_dict(orient="records"))


def load_and_process_data(excel_path, image_folder):
    df_raw = pd.read_excel(excel_path)

    #  Group by creative and sum metrics
    grouped = (
        df_raw.groupby("Creative")[["Impressions", "Clicks"]]
        .sum()
        .reset_index()
    )
    grouped["CTR"] = grouped["Clicks"] / grouped["Impressions"]
    grouped["campaign_name"] = grouped["Creative"]  # For consistent naming

    X = []
    y = []
    metadata = []

    for _, row in grouped.iterrows():
        image_path = os.path.join(image_folder, f"{row['campaign_name']}.jpg")
        if not os.path.exists(image_path):
            continue

        try:
            features = extract_clip_features(image_path)
            X.append(features)
            y.append(row["CTR"])
            metadata.append({
                "campaign_name": row["campaign_name"],
                "CTR": float(row["CTR"]),
                "image_path": image_path
            })
        except Exception as e:
            print(f"Error processing {row['campaign_name']}: {e}")

    return np.array(X), np.array(y), metadata

def train_model(X, y, metadata, save_path="model_store.npz"):
    model = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    for i in range(len(metadata)):
        metadata[i]["predicted_ctr"] = float(np.clip(preds[i], 0, 1))
        metadata[i]["embedding"] = X[i].astype(np.float32)
        metadata[i]["image_path"] = os.path.join("images", f"{metadata[i]['campaign_name']}.jpg")
    np.savez_compressed(save_path, metadata=metadata)
    joblib.dump(model, "ctr_model.pkl")

    return model, preds, r2

