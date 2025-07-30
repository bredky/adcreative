import pandas as pd
import re

def extract_creative_name(creative):
    try:
        match = re.search(r"Summer25-(.*?)-DE", str(creative))
        if match:
            return match.group(1).replace("-", " ").strip()
    except:
        pass
    return None

def main():
    # Step 1: Load the Excel file
    input_file = "Data/daily_Data.xlsx"
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    # Step 2: Extract creative names
    df["Creative Name"] = df["Creative"].apply(extract_creative_name)

    # Step 3: Filter unique non-null names
    unique_names = df["Creative Name"].dropna().unique()

    # Step 4: Save to new Excel file
    output_df = pd.DataFrame({"Creative Name": unique_names})
    output_file = "creative_pic.xlsx"
    try:
        output_df.to_excel(output_file, index=False)
        print(f"Saved {len(unique_names)} unique creative names to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    main()
