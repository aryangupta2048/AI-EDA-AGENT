
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Extract the file extension and convert to lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    try:
        if file_extension == ".csv":
            # You can add parameters like sep=',' or encoding='utf-8' if needed
            df = pd.read_csv(file_path)

        elif file_extension in [".xls", ".xlsx"]:
            # Requires 'openpyxl' for .xlsx or 'xlrd' for .xls
            df = pd.read_excel(file_path)

        elif file_extension == ".json":
            df = pd.read_json(file_path)

        elif file_extension == ".parquet":
            # Requires 'pyarrow' or 'fastparquet'
            df = pd.read_parquet(file_path)

        elif file_extension in [".txt", ".tsv"]:
            # Assuming tab-separated for .tsv or custom text
            sep = "\t" if file_extension == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)

        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        print(f"Successfully read {file_path} as a DataFrame.")
        return df

    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# --- Example Usage ---
# df = read_uploaded_file("path/to/your/uploaded_file.csv")
# print(df.head())
