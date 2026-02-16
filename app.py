import streamlit as st
from pathlib import Path
import traceback
import threading
import time
import logging

# Suppress Streamlit's bare-mode warning about missing ScriptRunContext
_sr_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
class _NoScriptRunContextWarn(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "missing ScriptRunContext" not in record.getMessage()
        except Exception:
            return True
_sr_logger.addFilter(_NoScriptRunContextWarn())

# Suppress general bare-mode warnings from Streamlit
_streamlit_logger = logging.getLogger("streamlit")
_ss_logger = logging.getLogger("streamlit.runtime.state.session_state_proxy")
class _SuppressBareModeWarn(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if "to view a Streamlit app on a browser" in msg:
                return False
            if "Session state does not function when running a script without`streamlit run`" in msg:
                return False
            if "Session state does not function when running a script without `streamlit run`" in msg:
                return False
            return True
        except Exception:
            return True
_streamlit_logger.addFilter(_SuppressBareModeWarn())
_ss_logger.addFilter(_SuppressBareModeWarn())

import run_experiment as re
import plot as plot_mod
from gate_library import GATE_LIBRARY


def parse_j_values(text: str) -> list[float]:
    vals = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(float(p) * 1e6)  # MHz → Hz
        except Exception:
            st.warning(f"Invalid J value: {p}")
    return vals


def coerce_alpha_value(val: float) -> float | int:
    v = round(float(val), 1)
    return int(v) if float(v).is_integer() else v
def coerce_alpha_list(text: str) -> list[float | int]:
    out = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = float(p)
            out.append(coerce_alpha_value(v))
        except Exception:
            st.warning(f"Invalid alpha entry: {p}")
    return out


def show_recent_plots(base_dir: Path):
    plots_root = base_dir
    if not plots_root.exists():
        return
    images = list(plots_root.glob("*.png"))
    if not images:
        return
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    st.subheader("Recent Plots")
    for img in images[:8]:
        st.image(str(img), caption=str(img.name))


def find_plot_images(
    base_dir: Path,
    gate: str | None = None,
    J_MHz: float | None = None,
    alpha_val: float | int | None = None,
    Joff_kHz: float | None = None,
    category: str | None = None,
    metric: str | None = None,
    noise_type: str | None = None,
    keywords: str | None = None,
):
    root_candidates = []
    if category == "Thresholds":
        # Search under cfg folders for both base and test_gates trees
        if gate and J_MHz:
            for root_base in [base_dir / "gates" / gate / f"J={int(J_MHz):.0f}MHz",
                              base_dir / "test_gates" / "gates" / gate / f"J={int(J_MHz):.0f}MHz"]:
                alpha_dirs: list[Path] = []
                if alpha_val is not None:
                    aval = round(float(alpha_val), 1)
                    alpha_main = str(int(aval)) if float(aval).is_integer() else f"{aval:.1f}"
                    alpha_dirs.append(root_base / f"alpha={alpha_main}")
                    if float(aval).is_integer():
                        alpha_dirs.append(root_base / f"alpha={aval:.1f}")
                else:
                    alpha_dirs = list(root_base.glob("alpha=*"))
                for ad in alpha_dirs:
                    joff_dirs = [ad / f"Joff={int(Joff_kHz):.0f}kHz"] if Joff_kHz is not None else list(ad.glob("Joff=*"))
                    for jd in joff_dirs:
                        for cfg in jd.glob("cfg_*"):
                            root_candidates.append(cfg / "Plots")
        else:
            root_candidates.extend((base_dir / "test_gates").glob("**/cfg_*/Plots"))
    else:
        if gate and J_MHz:
            gate_root_base = base_dir / "gates" / gate / f"J={int(J_MHz):.0f}MHz"
            alpha_dirs: list[Path] = []
            if alpha_val is not None:
                aval = round(float(alpha_val), 1)
                alpha_main = str(int(aval)) if float(aval).is_integer() else f"{aval:.1f}"
                alpha_dirs.append(gate_root_base / f"alpha={alpha_main}")
                if float(aval).is_integer():
                    alpha_dirs.append(gate_root_base / f"alpha={aval:.1f}")
            else:
                alpha_dirs = list(gate_root_base.glob("alpha=*"))

            if Joff_kHz is not None:
                for ad in alpha_dirs:
                    joff_dir = ad / f"Joff={int(Joff_kHz):.0f}kHz"
                    for cfg in joff_dir.glob("cfg_*"):
                        root_candidates.append(cfg / "Plots")
            else:
                for ad in alpha_dirs:
                    for sr in ad.glob("Joff=*"):
                        for cfg in sr.glob("cfg_*"):
                            root_candidates.append(cfg / "Plots")
        else:
            root_candidates.append(base_dir)

    images = []
    for rc in root_candidates:
        if not rc.exists():
            continue
        if category == "Noise":
            search_roots = [rc / "Noise"]
        elif category == "Heatmaps":
            # Support both runner plots (Plots/Clean) and legacy script plots (Plots/Deltat_DeltaV)
            search_roots = [rc / "Clean", rc / "Deltat_DeltaV"]
        elif category == "Jitter":
            search_roots = [rc / "Noise"]
        else:
            search_roots = [rc]

        for sr in search_roots:
            if not sr.exists():
                continue
            images.extend(sr.glob("**/*.png"))

    def match(p: Path) -> bool:
        name = p.name.lower()
        ok = True
        if metric:
            mkey = metric.lower().replace(" ", "_")
            ok = ok and (mkey in name)
        if category == "Noise" and noise_type:
            ok = ok and (noise_type in name)
        if category == "Jitter":
            ok = ok and ("jitter" in name)
        if category == "Heatmaps":
            ok = ok and ("heatmap" in name or "linear_pulse" in name or "rc_pulse" in name or "square_pulse" in name )
        if keywords:
            ok = ok and (keywords.lower() in name)
        return ok

    return [p for p in images if match(p)]


st.set_page_config(page_title="MEP Simulation UI", layout="wide")
st.title("MEP Simulation Control Panel")
st.write("Configure and run simulations, and generate plots from saved results.")

if "status_log" not in st.session_state:
    st.session_state["status_log"] = []
if "runner_thread" not in st.session_state:
    st.session_state["runner_thread"] = None
if "running" not in st.session_state:
    st.session_state["running"] = False

# Reflect actual thread alive state to avoid stale "running" indicator
try:
    th = st.session_state.get("runner_thread")
    if th is not None and hasattr(th, "is_alive"):
        if not th.is_alive():
            st.session_state["running"] = False
except Exception:
    pass

with st.sidebar:
    st.header("Configuration")
    base_dir_str = st.text_input("Base Results Directory", value=str(re.BASE_DIR))
    base_dir = Path(base_dir_str)

    force_eval = st.checkbox("Force Evaluation (new Results_vN)", value=bool(getattr(re, "FORCE_EVALUATION", False)))
    plot_only = st.checkbox("Plot Only", value=re.PLOT_ONLY)

    st.subheader("Run Sections")
    run_sections = {}
    for key in [
        "fidelities", 
        "heatmaps",
        "heatmaps_all",
        "jitter",
        "white_noise",
        "pink_noise",
        "noise",
    ]:
        run_sections[key] = st.checkbox(key, value=re.RUN.get(key, False))

    st.subheader("Sweep Parameters")
    gates_default = re.GATES if hasattr(re, "GATES") else ["X", "Y", "SXH"]
    gates = st.multiselect("Gates", options=list(GATE_LIBRARY.keys()), default=gates_default)
    j_values_text = st.text_input("J values (MHz, comma-separated)", value=",").strip() or ",".join([f"{v/1e6:.0f}" for v in getattr(re, "J_VALUES", [10e6, 20e6])])
    j_values = parse_j_values(j_values_text)

    n_jobs = st.number_input("Parallel jobs", min_value=1, max_value=64, value=getattr(re, "N_JOBS", 2))
    dt_ps = st.number_input("DT_PS (ps)", min_value=1, max_value=1000, value=getattr(re, "DT_PS", 15))
    iterations = st.number_input("Iterations per amplitude", min_value=1, max_value=10000, value=getattr(re, "iterations", 5))
    n_noise = st.number_input("Noise amplitude points (N_noise)", min_value=1, max_value=10000, value=getattr(re, "N_noise", 10))
    n_space = st.number_input("Heatmap grid size (N_space)", min_value=1, max_value=10000, value=getattr(re, "N_space", 25))

    alpha_base_input = st.number_input("alpha (base)", min_value=0.0, max_value=1000.0, value=float(getattr(re, "alpha", 25)), step=0.1)
    alpha_base = coerce_alpha_value(alpha_base_input)

    alpha_list_text = st.text_input("alpha list (comma)", value=",").strip() or ",".join([str(a) for a in getattr(re, "alpha_list", [25, 12.5])])
    alpha_list_vals = coerce_alpha_list(alpha_list_text)
    joff_list = st.text_input("Joffset list (kHz, comma)", value=",").strip() or ",".join([f"{a/1e3:.0f}" for a in getattr(re, "Joffset_list", [100e3, 10e3])])
    joff_list_vals = [float(v.strip())*1e3 for v in joff_list.split(",") if v.strip()]

    run_btn = st.button("Run Simulations")
    st.divider()
    st.subheader("Runner Control")
    stop_btn = st.button("Stop Run")
    refresh_status_btn = st.button("Refresh Status View")
    auto_refresh = st.checkbox("Auto-refresh status (1s)", value=True)
    st.subheader("Plot Browser")
    gate_sel = st.selectbox("Gate", options=[""] + list(GATE_LIBRARY.keys()), index=0)
    j_sel = st.text_input("J (MHz) for plot", value="")
    alpha_sel = st.text_input("alpha (optional)", value="")
    joff_sel = st.text_input("Joffset (kHz, optional)", value="")
    category = st.selectbox("Category", options=["Noise", "Jitter", "Heatmaps", "Thresholds"])
    metric = st.selectbox("Metric", options=["", "evolution fidelity", "state fidelity", "QPT fidelity"], index=0)
    noise_type = st.selectbox("Noise type", options=["", "white", "pink", "combined"], index=0)
    keywords = st.text_input("Filename keywords (optional)", value="")
    find_btn = st.button("Find Plots")

if run_btn and not st.session_state["running"]:
    try:
        re.BASE_DIR = base_dir
        re.FORCE_EVALUATION = force_eval
        re.PLOT_ONLY = plot_only
        for k, v in run_sections.items():
            re.RUN[k] = v
        re.GATES = gates
        re.J_VALUES = j_values if j_values else getattr(re, "J_VALUES", [])
        re.N_JOBS = int(n_jobs)
        re.DT_PS = int(dt_ps)
        re.iterations = int(iterations)
        re.N_noise = int(n_noise)
        re.N_space = int(n_space)
        # Preserve integer alpha when integral; else use one-decimal float
        re.alpha = alpha_base if isinstance(alpha_base, int) else float(alpha_base)
        re.alpha_list = [a for a in alpha_list_vals] if alpha_list_vals else getattr(re, "alpha_list", [])
        re.Joffset_list = joff_list_vals if joff_list_vals else getattr(re, "Joffset_list", [])
        # Reset and trim status log
        st.session_state["status_log"] = []
        # Avoid calling Streamlit APIs from background thread: use direct list appends
        re.STATUS_LOG = st.session_state["status_log"]
        def _cb(msg: str):
            re.STATUS_LOG.append(msg)
            # Trim to last 100 to bound memory
            if len(re.STATUS_LOG) > 100:
                # Slice in place to keep same list object reference
                del re.STATUS_LOG[:-100]
        re.STATUS_CALLBACK = _cb
        re.STOP_REQUESTED = False

        def _run():
            try:
                re.main()
            finally:
                # Do not call Streamlit APIs from background thread
                try:
                    re.status("Simulation finished.")
                except Exception:
                    try:
                        re.STATUS_LOG.append("Simulation finished.")
                    except Exception:
                        pass

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        st.session_state["runner_thread"] = th
        st.session_state["running"] = True
        st.success("Run started in background.")
    except Exception:
        st.error("Failed to start background run.")
        st.code(traceback.format_exc())

if stop_btn and st.session_state["running"]:
    try:
        re.STOP_REQUESTED = True
        st.warning("Stop requested. The run will end at the next safe checkpoint.")
    except Exception:
        st.error("Failed to send stop signal.")

if refresh_status_btn:
    st.info("Status refreshed.")

st.subheader("Status Log ")
if st.session_state["status_log"]:
    for line in st.session_state["status_log"][-5:]:
        st.write(line)
else:
    st.write("No status yet.")

# Show completion banner when not running and finished
if not st.session_state.get("running") and st.session_state.get("status_log"):
    last = st.session_state["status_log"][-1].lower()
    if "simulation finished" in last:
        st.success("Simulation finished.")

# Running indicator and optional auto-refresh
if st.session_state.get("running"):
    st.info("Simulation running…")
    if auto_refresh:
        # Only rerun when new log lines arrive or at interval
        if "_last_log_len" not in st.session_state:
            st.session_state["_last_log_len"] = 0
        current_len = len(st.session_state.get("status_log", []))
        time.sleep(1)
        st.session_state["_last_log_len"] = current_len
        try:
            # Prefer stable API when available
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
        except Exception:
            # If rerun is unavailable, silently skip auto-refresh
            pass
else:
    st.caption("Idle")

# Threshold plotting options removed per request

if find_btn:
    try:
        J_MHz = float(j_sel) if j_sel.strip() else None
        alpha_val = float(alpha_sel) if alpha_sel.strip() else None
        Joff_kHz = float(joff_sel) if joff_sel.strip() else None
        plots = find_plot_images(
            base_dir=base_dir,
            gate=(gate_sel or None),
            J_MHz=J_MHz,
            alpha_val=alpha_val,
            Joff_kHz=Joff_kHz,
            category=category,
            metric=(metric or None),
            noise_type=(noise_type or None),
            keywords=(keywords or None),
        )
        print(plots)
        if not plots:
            st.info("No plots found with the given filters.")
        else:
            st.success(f"Found {len(plots)} plot(s)")
            for p in plots[:12]:
                st.image(str(p), caption=str(p.name))
                st.code(str(p))
    except Exception:
        st.error("Error during plot search.")
        st.code(traceback.format_exc())

show_recent_plots(base_dir)
