import ast
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def get_project_root() -> Path:
    """
    Resolve the project root assuming this file lives at:
    code/reference_inception_model/hyperparameter_tuning/
    """
    # .../5utr_cnn/code/reference_inception_model/hyperparameter_tuning/3b_...
    # parents[0] = hyperparameter_tuning
    # parents[1] = reference_inception_model
    # parents[2] = code
    # parents[3] = 5utr_cnn  <-- project root
    return Path(__file__).resolve().parents[3]


def parse_parameters_column(param_str: str) -> dict:
    """
    Parse the 'parameters' string from summary.csv into a dict.

    Example:
    "batch_size=64, dropout_rate=0.05, filter_sizes=[3,5,7], l2_lambda=0.05, learning_rate=0.0005, ..."
    """
    # Strip surrounding quotes if present
    s = param_str.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # Parse key=value pairs, handling lists in brackets
    parsed = {}
    i = 0
    while i < len(s):
        # Skip whitespace and commas
        while i < len(s) and (s[i] in ' ,'):
            i += 1
        if i >= len(s):
            break
        
        # Find the key (everything up to '=')
        key_start = i
        while i < len(s) and s[i] != '=':
            i += 1
        if i >= len(s):
            break
        
        key = s[key_start:i].strip()
        i += 1  # Skip '='
        
        # Find the value (handle lists in brackets)
        val_start = i
        if i < len(s) and s[i] == '[':
            # This is a list, find the matching closing bracket
            bracket_count = 0
            while i < len(s):
                if s[i] == '[':
                    bracket_count += 1
                elif s[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        i += 1
                        break
                i += 1
        else:
            # Regular value, find the next comma (or end of string)
            while i < len(s) and s[i] != ',':
                i += 1
        
        val = s[val_start:i].strip()
        parsed[key] = val
    
    return parsed


def load_combo_architecture_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)

    # Parse the 'parameters' string into columns
    parsed_rows = []
    for _, row in df.iterrows():
        params = parse_parameters_column(row["parameters"])
        params["min_dev_loss"] = row["min_dev_loss"]
        
        # Convert filter_sizes from string to normalized list string
        if "filter_sizes" in params:
            try:
                # Parse the list string (e.g., "[3,5,7]" -> [3, 5, 7])
                filter_list = ast.literal_eval(params["filter_sizes"])
                # Normalize to sorted tuple string for consistent comparison
                # This ensures [3,5,7] and [5,3,7] are treated as the same
                params["filter_sizes"] = str(tuple(sorted(filter_list)))
            except (ValueError, SyntaxError):
                # If parsing fails, keep as-is
                pass
        
        parsed_rows.append(params)

    parsed_df = pd.DataFrame(parsed_rows)

    # Coerce core tuning columns to numeric where appropriate
    for col in ["n_conv_layers", "n_filters", "n_dense_layers", "n_dense_units", "min_dev_loss"]:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors="coerce")

    return parsed_df


def plot_all_dev_losses(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(df["min_dev_loss"].dropna(), bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    plt.xlabel("Min dev loss")
    plt.ylabel("Count")
    plt.title("Histogram of min dev losses (all combinations)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_dev_losses_by_param(df: pd.DataFrame, output_path: Path) -> None:
    """
    2x3 grid for architecture parameters:
      - n_conv_layers
      - n_filters
      - n_dense_layers
      - n_dense_units
      - filter_sizes

    Inside each of the 5 panels, create a stack of histograms,
    one for each value of the corresponding parameter.
    """
    params = ["n_conv_layers", "n_filters", "n_dense_layers", "n_dense_units", "filter_sizes"]

    fig = plt.figure(figsize=(18, 12))
    outer = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.25)

    for idx, param in enumerate(params):
        if param not in df.columns:
            continue

        row, col = divmod(idx, 3)
        
        # Handle filter_sizes specially (it's a list string)
        if param == "filter_sizes":
            # Convert string representations to a consistent format for grouping
            unique_vals = sorted(df[param].dropna().unique(), key=lambda x: str(x))
        else:
            unique_vals = sorted(df[param].dropna().unique())
        
        if len(unique_vals) == 0:
            continue

        # Create a stack inside this outer cell (one histogram per unique value)
        inner = outer[row, col].subgridspec(len(unique_vals), 1, hspace=0.1)

        for j, val in enumerate(unique_vals):
            ax = fig.add_subplot(inner[j, 0])
            mask = df[param] == val
            ax.hist(
                df.loc[mask, "min_dev_loss"].dropna(),
                bins=15,
                color="steelblue",
                edgecolor="black",
                alpha=0.8,
            )

            # Only put the main title once per parameter (on the top subplot)
            if j == 0:
                ax.set_title(f"{param}", fontsize=12)

            # Label y-axis with the specific value to save space
            # For filter_sizes, convert tuple string back to list format for display
            if param == "filter_sizes":
                try:
                    # Convert "(3, 5, 7)" back to "[3,5,7]" for display
                    filter_tuple = ast.literal_eval(val)
                    val_label = "[" + ",".join(str(x) for x in filter_tuple) + "]"
                except (ValueError, SyntaxError):
                    val_label = str(val).replace("(", "[").replace(")", "]")
            else:
                val_label = str(val)
            ax.set_ylabel(val_label, fontsize=9)

            # Only show x-label on the bottom subplot of each stack
            if j == len(unique_vals) - 1:
                ax.set_xlabel("Min dev loss")
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_ordered_summary(df: pd.DataFrame, original_summary_path: Path, output_path: Path) -> None:
    """
    Sort combinations from lowest dev loss to highest and write to summary_ordered.csv.
    Keep the original 'parameters' string plus parsed columns for convenience.
    """
    original_df = pd.read_csv(original_summary_path)
    parsed_df = df.copy()

    # Ensure alignment: same order as original, then join on min_dev_loss index
    parsed_df = parsed_df.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)

    combined = pd.concat([original_df, parsed_df.drop(columns=["min_dev_loss"], errors="ignore")], axis=1)
    combined["min_dev_loss"] = parsed_df["min_dev_loss"]

    combined_sorted = combined.sort_values("min_dev_loss", ascending=True)
    combined_sorted.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    project_root = get_project_root()

    # Where the inception combo-architecture summary currently lives
    inception_summary_path = (
        project_root
        / "output"
        / "reference_inception_model"
        / "hyperparameter_tuning"
        / "combo_architecture"
        / "summary.csv"
    )

    if not inception_summary_path.exists():
        raise FileNotFoundError(f"Could not find summary.csv at {inception_summary_path}")

    # Where to write the requested outputs
    # (same directory tree as the input summary, per user request)
    output_dir = (
        project_root
        / "output"
        / "reference_inception_model"
        / "hyperparameter_tuning"
        / "combo_architecture"
    )
    os.makedirs(output_dir, exist_ok=True)

    df = load_combo_architecture_summary(inception_summary_path)

    # 1) Histogram of all dev losses
    hist_all_path = output_dir / "combo_architecture_dev_loss_hist_all.png"
    plot_all_dev_losses(df, hist_all_path)

    # 2) Histogram split by the 5 architecture parameters (2x3)
    hist_by_param_path = output_dir / "combo_architecture_dev_loss_hist_by_param.png"
    plot_dev_losses_by_param(df, hist_by_param_path)

    # 3) Ordered list of combinations
    summary_ordered_path = output_dir / "summary_ordered.csv"
    write_ordered_summary(df, inception_summary_path, summary_ordered_path)


if __name__ == "__main__":
    main()

