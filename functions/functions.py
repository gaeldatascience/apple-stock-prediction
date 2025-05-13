from datetime import datetime
import pandas as pd
import numpy as np
import os


def set_plot_style():
    import plotly.io as pio
    import seaborn as sns
    import matplotlib.pyplot as plt

    """
    Set the style for matplotlib and plotly plots.
    """
    pio.templates.default = "plotly_white"

    sns.set_theme(style="whitegrid", palette="muted")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.edgecolor": "white",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.color": "grey",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (12, 6),
            "xtick.bottom": False,
            "ytick.left": False,
        }
    )


def import_and_preprocess_data_stock():
    """
    Import and preprocess the stock data from a parquet file.
    """

    data = pd.read_parquet("data/AAPL_data.pq")
    data = data.drop(columns=["index"])
    data["Volatility"] = data["High"] - data["Low"]
    data["Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log1p(data["Return"])
    return data


def import_and_preprocess_data_tweets():
    """
    Import and preprocess the tweets data from a parquet file.
    """

    data = pd.read_parquet("data/tweets_aggregated.pq")
    data["created_at"] = pd.to_datetime(data["created_at"]).dt.strftime("%Y-%m-%d")
    data = data.drop(columns=["Unnamed: 0"]).rename(columns={"created_at": "date"})
    return data


def save_arch_results_to_latex(
    results, p=None, q=None, folder_path="tab", float_format="%.4f", model_type="GARCH"
):
    """
    Save ARCH or GARCH model results as a LaTeX table.
    """

    model_type = model_type.upper()

    # Factory configuration
    model_config = {
        "ARCH": {
            "filename": lambda p, q: f"arch_{p}_results.tex",
            "caption": lambda p, q: f"ARCH({p}) Parameter Estimates",
            "label": lambda p, q: f"tab:arch_{p}",
        },
        "GARCH": {
            "filename": lambda p, q: f"garch_{p}_{q}_results.tex",
            "caption": lambda p, q: f"GARCH({p},{q}) Parameter Estimates",
            "label": lambda p, q: f"tab:garch_{p}_{q}",
        },
    }

    config = model_config[model_type]

    # Generate file info
    filename = config["filename"](p, q)
    caption = config["caption"](p, q)
    label = config["label"](p, q)
    full_path = os.path.join(folder_path, filename)

    # Create LaTeX table from model summary
    summary_df = pd.DataFrame(
        {
            "coef": results.params,
            "std err": results.std_err,
            "t": results.tvalues,
            "P>|t|": results.pvalues,
        }
    )

    latex_code = summary_df.to_latex(
        column_format="lcccc",
        float_format=lambda x: float_format % x,
        caption=caption,
        label=label,
        escape=False,
    )

    os.makedirs(folder_path, exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
