from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np

mcp = FastMCP("Data MCP Server")

@mcp.tool()
def profile_dataset(file_path: str) -> dict:
    """Profile dataset to extract summary statistics, missing values, and outliers."""
    df = pd.read_csv(file_path)
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }
    return summary

@mcp.tool()
def detect_imbalance_and_outliers(file_path: str, target_column: str) -> dict:
    """Analyze class imbalance and statistical outliers."""
    df = pd.read_csv(file_path)
    imbalance = df[target_column].value_counts(normalize=True).to_dict() if target_column in df else {}
    
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
    outliers_count = int((z_scores > 3).sum().sum())
    
    return {
        "class_imbalance": imbalance,
        "outliers_detected": outliers_count
    }

if __name__ == "__main__":
    mcp.run()
