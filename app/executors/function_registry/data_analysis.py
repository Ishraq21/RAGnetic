# app/executors/function_registry/data_analysis.py
import base64
import io
import logging
import re
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .__init__ import FunctionRegistration

logger = logging.getLogger(__name__)


# --- Shared Helpers ---

def _read_csv_safe(file_path: str, delimiter: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Helper to safely read a CSV file with robust error handling."""
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found at '{file_path}'.")
    except UnicodeDecodeError as e:
        logger.warning(f"Encoding error with {encoding}, retrying with latin1: {e}")
        try:
            return pd.read_csv(file_path, delimiter=delimiter, encoding="latin1")
        except Exception as e2:
            logger.error(f"Fallback encoding also failed: {e2}")
            raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file at '{file_path}': {e}")
        raise ValueError(f"Failed to parse CSV: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading CSV: {e}")
        raise RuntimeError(f"Unexpected error reading file: {e}")


# --- Function Definitions ---

@FunctionRegistration(
    name="get_dataframe_info",
    description="Reads a CSV file and returns a summary including the first N rows, column names, and data types for initial inspection.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "n": {"type": "integer", "description": "Number of rows to return for the preview.", "default": 5},
            "delimiter": {"type": "string", "description": "CSV delimiter character.", "default": ","},
            "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"}
        },
        "required": ["file_path"]
    }
)
def get_dataframe_info(file_path: str, n: int = 5, delimiter: str = ",", encoding: str = "utf-8") -> Dict[str, Any]:
    """Returns a summary of a CSV file, including a head preview and column information."""
    try:
        df = _read_csv_safe(file_path, delimiter, encoding)
        return {
            "summary": f"Initial preview of the dataframe from '{file_path}'.",
            "head": df.head(n).to_dict(orient='records'),
            "total_records": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict()
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="generate_descriptive_statistics",
    description="Generates descriptive statistics (count, mean, std, min, max, etc.) for all numerical columns in a CSV file.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "delimiter": {"type": "string", "description": "CSV delimiter character.", "default": ","},
            "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"}
        },
        "required": ["file_path"]
    }
)
def generate_descriptive_statistics(file_path: str, delimiter: str = ",", encoding: str = "utf-8") -> Dict[str, Any]:
    """Generates descriptive statistics for all numerical columns."""
    try:
        df = _read_csv_safe(file_path, delimiter, encoding)
        stats = df.describe(include=[np.number]).to_dict()
        return {
            "summary": "Descriptive statistics generated.",
            "statistics": stats,
            "records_processed": len(df),
            "columns_analyzed": list(df.select_dtypes(include=[np.number]).columns)
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="get_value_distribution",
    description="Calculates the frequency distribution for a given column in a CSV. Returns the top 10 most frequent values.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "column_name": {"type": "string", "description": "The name of the column to analyze."},
            "delimiter": {"type": "string", "description": "CSV delimiter character.", "default": ","},
            "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"}
        },
        "required": ["file_path", "column_name"]
    }
)
def get_value_distribution(file_path: str, column_name: str, delimiter: str = ",", encoding: str = "utf-8") -> Dict[
    str, Any]:
    """Calculates frequency counts for a column."""
    try:
        df = _read_csv_safe(file_path, delimiter, encoding)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        distribution = df[column_name].value_counts().nlargest(10).to_dict()
        return {
            "summary": f"Top 10 value distributions for column '{column_name}'.",
            "distribution": distribution,
            "total_records": len(df),
            "columns_analyzed": [column_name]
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="analyze_correlation_heatmap",
    description="Calculates correlations between all numerical columns and returns a heatmap image (base64 PNG).",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "delimiter": {"type": "string", "description": "CSV delimiter character.", "default": ","},
            "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"}
        },
        "required": ["file_path"]
    }
)
def analyze_correlation_heatmap(file_path: str, delimiter: str = ",", encoding: str = "utf-8") -> Dict[str, Any]:
    """Generates a correlation heatmap for numerical columns."""
    try:
        df = _read_csv_safe(file_path, delimiter, encoding)
        corr = df.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))
        plt.imshow(corr, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=90)
        plt.yticks(range(len(corr)), corr.columns)
        plt.title("Correlation Heatmap")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close('all')

        return {
            "summary": "Correlation heatmap generated.",
            "image_base64_png": img_str,
            "correlations": corr.to_dict(),
            "columns_analyzed": list(corr.columns)
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="get_top_n_by_column",
    description="Returns the top N records sorted by a specified numerical column, in either ascending or descending order.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "column_name": {"type": "string", "description": "The numerical column to sort by."},
            "n": {"type": "integer", "description": "Number of records to return.", "default": 10},
            "ascending": {"type": "boolean", "description": "Sort order (true for ascending, false for descending).",
                          "default": False},
            "delimiter": {"type": "string", "description": "CSV delimiter character.", "default": ","},
            "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"}
        },
        "required": ["file_path", "column_name"]
    }
)
def get_top_n_by_column(file_path: str, column_name: str, n: int = 10, ascending: bool = False,
                        delimiter: str = ",", encoding: str = "utf-8") -> Dict[str, Any]:
    """Returns the top N records sorted by a numerical column."""
    try:
        df = _read_csv_safe(file_path, delimiter, encoding)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        # Ensure the column is numerical before sorting
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise TypeError(f"Column '{column_name}' is not numerical and cannot be sorted.")

        top_records = df.sort_values(by=column_name, ascending=ascending).head(n)
        return {
            "summary": f"Top {n} records by '{column_name}'.",
            "data": top_records.to_dict(orient="records"),
            "records_processed": len(df)
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="group_by_aggregation",
    description="Performs sum, mean, and count aggregations on a target column, grouped by a specified categorical column.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "group_by_column": {"type": "string", "description": "The categorical column to group by."},
            "target_column": {"type": "string", "description": "The numerical column to aggregate."},
            "delimiter": {"type": "string", "description": "CSV delimiter character.", "default": ","},
            "encoding": {"type": "string", "description": "File encoding.", "default": "utf-8"}
        },
        "required": ["file_path", "group_by_column", "target_column"]
    }
)
def group_by_aggregation(file_path: str, group_by_column: str, target_column: str,
                         delimiter: str = ",", encoding: str = "utf-8") -> Dict[str, Any]:
    """Performs aggregations grouped by a specified column."""
    try:
        df = _read_csv_safe(file_path, delimiter, encoding)
        if group_by_column not in df.columns or target_column not in df.columns:
            raise ValueError("One or both columns not found.")
        agg_df = df.groupby(group_by_column)[target_column].agg(['sum', 'mean', 'count']).reset_index()
        return {
            "summary": f"Aggregations for '{target_column}' grouped by '{group_by_column}'.",
            "data": agg_df.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="perform_linear_regression",
    description="Performs a simple linear regression to predict a target variable based on a single independent variable. Returns the R² score and coefficients.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "independent_variable": {"type": "string", "description": "The independent variable column name."},
            "target_variable": {"type": "string", "description": "The target variable column name."}
        },
        "required": ["file_path", "independent_variable", "target_variable"]
    }
)
def perform_linear_regression(file_path: str, independent_variable: str, target_variable: str) -> Dict[str, Any]:
    """Performs linear regression and returns the R² score and coefficients."""
    try:
        df = _read_csv_safe(file_path)

        if independent_variable not in df.columns or target_variable not in df.columns:
            raise ValueError("Independent or target variable column not found.")

        X = df[[independent_variable]].dropna()
        y = df[target_variable].loc[X.index].dropna()

        if X.empty or y.empty:
            raise ValueError("Data is empty after dropping missing values.")

        model = LinearRegression().fit(X, y)
        score = model.score(X, y)

        return {
            "summary": f"Linear regression performed: '{target_variable}' predicted by '{independent_variable}'.",
            "r2_score": score,
            "coefficient": model.coef_[0],
            "intercept": model.intercept_
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="perform_cluster_analysis",
    description="Performs K-Means clustering on numerical data to group similar records. Returns cluster labels and a quality score.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "n_clusters": {"type": "integer", "description": "The number of clusters to form.", "default": 3}
        },
        "required": ["file_path"]
    }
)
def perform_cluster_analysis(file_path: str, n_clusters: int = 3) -> Dict[str, Any]:
    """Performs K-Means clustering on numerical data."""
    try:
        df = _read_csv_safe(file_path)
        numeric_df = df.select_dtypes(include=np.number).dropna()

        if numeric_df.empty:
            raise ValueError("No numerical data available for clustering.")
        if len(numeric_df) < n_clusters:
            raise ValueError("Not enough records to form the requested number of clusters.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(scaled_data)

        # Calculate silhouette score for cluster quality if there are multiple clusters
        score = silhouette_score(scaled_data, kmeans.labels_) if n_clusters > 1 else None

        # Add cluster labels to the original dataframe for context
        numeric_df['cluster_label'] = kmeans.labels_

        return {
            "summary": f"K-Means clustering performed with {n_clusters} clusters.",
            "cluster_labels": numeric_df['cluster_label'].to_dict(),
            "silhouette_score": score,
            "records_processed": len(numeric_df)
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@FunctionRegistration(
    name="perform_sentiment_analysis",
    description="Performs sentiment analysis on a specified text column in a CSV file and returns the average sentiment polarity.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file in the sandbox."},
            "text_column": {"type": "string", "description": "The name of the column containing text data to analyze."},
        },
        "required": ["file_path", "text_column"]
    }
)
def perform_sentiment_analysis(file_path: str, text_column: str) -> Dict[str, Any]:
    """Performs sentiment analysis on a text column."""
    try:
        df = _read_csv_safe(file_path)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in the CSV file.")

        sentiments = df[text_column].dropna().apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        return {
            "summary": f"Sentiment analysis performed on column '{text_column}'.",
            "average_sentiment_polarity": sentiments.mean(),
            "records_analyzed": len(sentiments)
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}