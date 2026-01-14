import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

def auto_plot(filename, base_dir="results_cadence"):
    """
    Auto-detect CSV format and plot:
    1) Delay vs digital code (DTC)
    2) Delay vs VDD
    3) Power vs capacitance

    filename can be:
    - "results_delay.csv"
    - "results_cadence/results_delay.csv"
    """
    path = Path(filename)
    if not path.exists():
        path = Path(base_dir) / filename
    
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    cols = df.columns

    name = path.stem

    # ============================================================
    # FORMAT 1: DTC delay vs digital code (b2 + delay b0/b1)
    # ============================================================
    if "b2" in cols and any(c.startswith("delay") for c in cols):
        data = []

        for _, row in df.iterrows():
            b2 = int(row["b2"])
            for col in cols:
                m = re.search(r"b0\s*(\d)\s*b1\s*(\d)", col)
                if not m:
                    continue
                b0, b1 = int(m.group(1)), int(m.group(2))
                code = b2 * 4 + b1 * 2 + b0
                data.append((code, row[col]))

        plot_df = (
            pd.DataFrame(data, columns=["code", "delay"])
            .sort_values("code")
            .reset_index(drop=True)
        )

        plt.figure()
        plt.plot(plot_df["code"], plot_df["delay"], marker="o")
        plt.xlabel("Digital code")
        plt.ylabel("Delay [s]")
        plt.title(name)
        plt.grid(True)
        plt.show()

        return plot_df

    # ============================================================
    # FORMAT 2 & 3: generic X–Y tables (delay vs VDD, power vs C)
    # ============================================================
    x_cols = [c for c in cols if c.endswith("X")]
    y_cols = [c for c in cols if c.endswith("Y")]

    if len(x_cols) == 1 and len(y_cols) == 1:
        x, y = x_cols[0], y_cols[0]

        plt.figure()
        plt.plot(df[x], df[y], marker="o")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(name)
        plt.grid(True)
        plt.show()

        return df[[x, y]]

    # ============================================================
    # Unknown format
    # ============================================================
    raise ValueError("Unknown CSV format – cannot auto-plot.")



def plot_csv_signals(filename,base_dir="results_cadence"):
    """
    Reads a CSV file with paired X and Y columns for different signals and plots them.
    """
    path = Path(filename)
    if not path.exists():
        path = Path(base_dir) / filename
    # Load the data
    df = pd.read_csv(path)
    
    # Identify signal groups by finding all columns ending in ' X'
    x_cols = [col for col in df.columns if col.endswith(' X')]
    
    plt.figure(figsize=(12, 7))
    
    for x_col in x_cols:
        # Construct the corresponding Y column name
        y_col = x_col[:-2] + ' Y' 
        
        if y_col in df.columns:
            # Convert columns to numeric, coercing strings/spaces to NaN
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            
            # Remove NaN values to ensure a clean plot
            mask = x_data.notna() & y_data.notna()
            x_plot = x_data[mask]
            y_plot = y_data[mask]
            
            # Create a label by removing the leading slash and ' X' suffix
            label = x_col[:-2].lstrip('/')
            
            plt.plot(x_plot, y_plot, label=label)
            
    plt.xlabel('Time / X-units')
    plt.ylabel('Voltage / Y-units')
    plt.title(f'Signal Analysis: {Path(filename).name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_filename = 'signals_plot.png'
    plt.savefig(output_filename)
    return output_filename

# Usage
plot_csv_signals('plot_signals.csv')

auto_plot("delay_vs_vdd.csv")
auto_plot("delay_vs_C.csv")
auto_plot("delay_vs_inverter_strength.csv")

auto_plot("power_vs_C.csv")
auto_plot("power_vs_ivdd.csv")
auto_plot("power_vs_inverter_strength.csv")


