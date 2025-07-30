import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import joblib
import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
from prompts import generate_pros_cons_prompt
from dotenv import load_dotenv
load_dotenv(".env.local")
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



from model.train_model import load_and_process_data, train_model
from clip_model.clip_extractor import extract_clip_features

st.set_page_config(page_title="CTR Predictor", layout="wide")
st.title(" Ad Creative CTR Predictor")

# Sidebar
st.sidebar.title(" Upload Campaign Data")
excel_file = st.sidebar.file_uploader("Upload your campaign metrics Excel file", type=["xlsx"])

image_folder = st.sidebar.text_input("Path to your image folder", value="images/")

import re

def get_image_path(creative_name):
    image_path = os.path.join("images", f"{creative_name}.jpg")
    return image_path if os.path.exists(image_path) else None

def clean_gpt_code(gpt_code):
    # Remove code fences and 'python' artifacts
    code = gpt_code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)  # Remove starting ```python or ```
    code = re.sub(r"\s*```$", "", code)            # Remove trailing ```
    code = code.strip()
    return code

# === Helper function to extract taxonomy values ===
def extract_taxonomy_value(text, key):
    try:
        parts = str(text).split('_')
        for part in parts:
            if part.startswith(f"{key}~"):
                return part.split('~')[1].strip()
        return None
    except:
        return None

# === Enrich dataframe with derived fields ===
def enrich_dataframe(df):
    if "Campaign" in df.columns:
        df["Objective"] = df["Campaign"].apply(lambda x: extract_taxonomy_value(x, "CA"))
        df["Project"] = df["Campaign"].apply(lambda x: extract_taxonomy_value(x, "MB"))
    if "Ad" in df.columns:
        df["Size"] = df["Ad"].apply(lambda x: extract_taxonomy_value(x, "SZ"))
        df["Language"] = df["Ad"].apply(lambda x: extract_taxonomy_value(x, "LG"))
        df["Market"] = df["Ad"].apply(lambda x: extract_taxonomy_value(x, "MK"))
        df["Channel"] =df["Ad"].apply(lambda x: extract_taxonomy_value(x, "CH"))

    df = df.iloc[:-1]
    return df

def get_column_summary(df):
    summary = {}
    for col in ["Campaign", "Creative", "Objective", "Project", "Date", "Ad" , "Language", "Market", "Channel", "Size", "Site (CM360)"]:
        if col in df.columns:
            unique_vals = df[col].dropna().astype(str).unique().tolist()
            summary[col] = unique_vals[:10]  # show top 20 per column
    return summary

def query_chatbot(df, user_prompt):
    system_prompt = f"""
You are a helpful data assistant. You are working with a Pandas dataframe called `df` with the following columns:
{list(df.columns)}

Here are sample values for key columns:
{get_column_summary(df)}

Your job is to:
1. Generate **pure Python code** using Pandas to answer the user's question.
2. Assign the result to a variable called `result`.
3. On a new line, provide a chart type comment: e.g., `# chart: bar`, `# chart: line`, or `# chart: none`.
when grouping creatives, remember that there can be multiple rows for the same creative and date, so make sure to aggregate them as together when required
Only suggest a chart if the result is a DataFrame or Series (e.g. grouped output, daily trend, comparison).
If the result is a single number or scalar, return `# chart: none`.
If your result is a DataFrame with one or more numeric columns and one categorical column, use .set_index() to make the categorical column the x-axis. You dont have to do this when not needed, only do this if you think its needed to make more sense
If the data contains multiple rows for the same value (e.g. same Creative, Campaign, or Date), use .groupby() and aggregate (e.g. .sum() or .mean()) before assigning to result. Do not return raw repeated rows unless specifically requested.”
Do not import anything. Only use variables `df` and `pd`.
Return only code and the chart comment. No explanation.

✅ Additionally: if the query involves any Creative(s), return a second variable called `creative_info` that contains a grouped summary of the matching creative(s) from the original dataframe `df`. make sure that it is filtered and is respective to the result only
make sure to group them respectively, i want impressions, clicks, click rate, size, market, language, channel, objective, project. Do NOT call .set_index("Creative") after a .groupby("Creative") aggregation — it's already the index.
Only use .set_index("Creative") if the Creative column was reset or not the index.




    """.strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )

    code = response.choices[0].message.content.strip()

    return code

def parse_code_and_chart_type(gpt_code):
    lines = gpt_code.strip().splitlines()
    chart_type = "none"
    code_lines = []

    for line in lines:
        if line.strip().lower().startswith("# chart:"):
            chart_type = line.strip().split(":", 1)[1].strip().lower()
        else:
            code_lines.append(line)

    return "\n".join(code_lines), chart_type


# Tabs
tab1, tab2, tab3 = st.tabs([" View Predictions", " Predict New Ad", "Query Exisiting Data"])

# TAB 1: View Predictions
with tab1:
    if excel_file and image_folder:
        st.info("Processing data and extracting image features...")
        raw_df = pd.read_excel(excel_file)
        raw_df.columns = [col.strip() for col in raw_df.columns]

        grouped = raw_df.groupby("Creative", as_index=False).agg({
            "Impressions": "sum",
            "Clicks": "sum"
        })

        grouped["CTR"] = grouped["Clicks"] / grouped["Impressions"]
        grouped = grouped.rename(columns={"Creative": "campaign_name"})

        os.makedirs("data", exist_ok=True)
        processed_path = "data/clean_grouped_metrics.xlsx"
        grouped.to_excel(processed_path, index=False)
        st.markdown("### grouped metrics")
        st.dataframe(grouped)

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


        if len(X) == 0:
            st.error("No valid images found matching the campaign names.")
        else:
            model, raw_preds, r2 = train_model(X, y, df)

            
            preds = np.clip(raw_preds, 0, 1)

            df["Predicted CTR"] = np.round(preds, 4)
            df["Actual CTR"] = np.round(y, 4)
            df["CTR Error"] = np.round(np.abs(df["Predicted CTR"] - df["Actual CTR"]), 4)

            
            df["Image Path"] = df["campaign_name"].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

            st.success(f"Model trained! R² Score: **{r2:.4f}**")

            st.subheader(" CTR Predictions")

            st.markdown("### Actual vs Predicted CTR")
            chart_df = df[["campaign_name", "Actual CTR", "Predicted CTR"]].set_index("campaign_name")
            st.bar_chart(chart_df)
            for _, row in df.iterrows():
                col1, col2 = st.columns([1, 3])
                image_path = os.path.join(image_folder, f"{row['campaign_name']}.jpg")

                if os.path.exists(image_path):
                    col1.image(Image.open(image_path), width=150)

                col2.markdown(f"""
                **Campaign:** `{row['campaign_name']}`  
                -  **Actual CTR:** `{row['CTR']:.4f}`  
                - **Predicted CTR:** `{row['Predicted CTR']:.4f}`  
                - **CTR Error:** `{row['CTR Error']:.4f}`
                """)
                st.markdown("---")
                

with tab2:
    st.subheader("Upload a New Ad Image")
    uploaded_image = st.file_uploader("Upload JPG or PNG", type=["jpg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Creative", use_column_width=True)

        try:
            # Extract CLIP features
            temp_path = "temp_uploaded.jpg"
            image.save(temp_path)
            features = extract_clip_features(temp_path)

            if "model" not in locals():
                st.warning(" Please train a model first in Tab 1.")
            else:
                pred_ctr = float(np.clip(model.predict([features])[0], 0, 1))
                st.success(f" Predicted CTR: **{pred_ctr:.4f}**")

                openai.api_key = os.getenv("OPENAI_API_KEY")
                prompt = generate_pros_cons_prompt(pred_ctr)

                from utils.vision import encode_image_to_base64

                # Encode image for GPT-4-Vision
                base64_image = encode_image_to_base64(temp_path)

                with st.spinner("Analyzing ad..."):
                    chat_completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": base64_image,
                                            "detail": "high"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": f"""Analyze this ad creative based on the following criteria:
                - Visual appeal and design clarity
                - Message clarity and strength of call-to-action
                - Storytelling or outcome demonstration
                - Use of trust-building elements (reviews, press)
                - Brand visibility and identity
                - Relevance to audience or timing
                - Alignment with CTR and conversion performance

                The predicted CTR is {pred_ctr:.4f}.

                Return 2 bullet points for:
                - Pros (what works)
                - Cons (what could be improved)
                """
                                    }
                                ]
                            }
                        ],
                        max_tokens=500
                    )

                feedback = chat_completion.choices[0].message.content
                st.markdown("### 🤖 GPT-4 Vision Feedback")
                st.markdown(feedback)


                # similarity searchcZ
                st.markdown("###  Top 3 Similar Ads")
                db = np.load("model_store.npz", allow_pickle=True)["metadata"]
                db = db.tolist()

                emb_matrix = np.array([entry["embedding"] for entry in db])
                sim_scores = cosine_similarity([features], emb_matrix)[0]
                top_idxs = sim_scores.argsort()[::-1][:3]

                for idx in top_idxs:
                    entry = db[idx]
                    col1, col2 = st.columns([1, 3])
                    if os.path.exists(entry["image_path"]):
                        col1.image(entry["image_path"], width=150)
                    col2.markdown(f"""
                    **{entry['campaign_name']}**  
                    -  Predicted CTR: `{entry['predicted_ctr']:.4f}`  
                    - Actual CTR: `{entry['actual_ctr']:.4f}`
                    """)
                    st.markdown("---")

        except Exception as e:
            st.error(f" Error processing image: {e}")


with tab3:
    st.header("Campaign Query Tool")

    uploaded_file = st.file_uploader("Upload Daily Campaign Delivery File", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = enrich_dataframe(df)

        st.session_state["raw_campaign_df"] = df

        st.subheader("Data Preview")
        st.dataframe(df.head(20))
    else:
        st.info("Please upload a daily delivery file to begin.")

    st.subheader(" Ask Questions About Your Daily Data")
    user_question = st.text_input("Ask a question eg- How many clicks did creative Engagement-Display-Summer25-Inspire-TLP-DE-Grn-300x250-NA get on July 24")

    if user_question and "raw_campaign_df" in st.session_state:
        df = st.session_state["raw_campaign_df"]

        try:
            gpt_code = query_chatbot(df, user_question)
            clean_code_raw = clean_gpt_code(gpt_code)
            clean_code, chart_type = parse_code_and_chart_type(clean_code_raw)
            st.code(clean_code, language="python")

            try:
                local_vars = {
                    "df": df.copy(),
                    "pd": pd,
                    "np": np
                }
                local_vars["df"]["Date"] = pd.to_datetime(local_vars["df"]["Date"])
                exec(clean_code, {}, local_vars)
                result = local_vars.get("result", "No result returned.")
                creative_info = local_vars.get("creative_info", None)
            except Exception as e:
                st.error(f" Error running cleaned code:\n{e}")

            if isinstance(result, (int, float, str, np.integer, np.floating)):
            
                result = result.item() if isinstance(result, np.generic) else result
                st.metric(label="Result", value=result)
            else:
                st.dataframe(result)
                    # Optional chart rendering based on GPT suggestion
                if chart_type != "none" and isinstance(result, (pd.DataFrame, pd.Series)):
                    st.markdown("####  Suggested Chart")
                    try:
                        if chart_type == "line":
                            st.line_chart(result)
                        elif chart_type == "bar":
                            st.bar_chart(result)
                        elif chart_type == "area" or chart_type == "stacked":
                            st.area_chart(result)
                        elif chart_type == "pie":
                            st.pyplot(result.plot.pie(autopct="%1.1f%%", legend=False).figure)
                        elif chart_type == "scatter":
                            st.scatter_chart(result)
                    except Exception as e:
                        st.error(f" Error rendering chart: {e}")
            if isinstance(creative_info, pd.DataFrame) and not creative_info.empty:
                st.markdown("#### 🎨 Creative Summary")
                creative_info["Image"] = creative_info.index.map(get_image_path)

                for idx, row in creative_info.iterrows():
                    col1, col2 = st.columns([1, 3])

                    if row["Image"]:
                        col1.image(row["Image"], width=150)
                    else:
                        col1.write("❌ No image")

                    col2.markdown("\n".join([
                        f"**{col}:** {row[col]}" for col in creative_info.columns if col != "Image"
                    ]))


        except Exception as e:
            st.error(f"Error while running GPT-generated code:\n{e}")