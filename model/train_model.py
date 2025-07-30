import pandas as pd
import numpy as np
from clip_model.clip_extractor import extract_clip_features
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import os
import joblib

def load_and_process_data(excel_path, image_folder):
    df_raw = pd.read_excel(excel_path)

    df_raw.columns = [col.strip() for col in df_raw.columns]

    grouped = df_raw.groupby("Creative").agg({
        "Impressions": "sum",
        "Clicks": "sum"
    }).reset_index()

    
    grouped["CTR"] = grouped["Clicks"] / grouped["Impressions"]
    grouped["campaign_name"] = grouped["Creative"]  

    X = []
    y = []
    valid_rows = []

    for _, row in grouped.iterrows():
        image_path = os.path.join(image_folder, f"{row['campaign_name']}.jpg")
        if os.path.exists(image_path):
            try:
                features = extract_clip_features(image_path)
                X.append(features)
                y.append(row["CTR"])
                valid_rows.append(row)
            except Exception as e:
                print(f"Skipping {row['campaign_name']} due to error: {e}")
        else:
            print(f"Image not found for {row['campaign_name']}")

    df = pd.DataFrame(valid_rows)
    return np.array(X), np.array(y), df

def train_model(X, y, df, save_path="model_store.npz"):
    model = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)


    df["predicted_ctr"] = np.clip(preds, 0, 1)
    df["embedding"] = list(X)
    
    np.savez_compressed(save_path, metadata=df.to_dict(orient="records"))
    joblib.dump(model, "ctr_model.pkl")

    return model, preds, r2
