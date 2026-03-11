
import json
import os
import sys
import glob

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.visualizers import (
    training_reward_curves,
    loss_curve,
    traffic_heatmap,
    trading_candlestick,
    supply_chain_flow,
    elo_leaderboard,
    experiment_comparison,
)

st.set_page_config(
    page_title="MARL Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,
[data-testid="stSidebar"] { background: rgba(15,23,42,0.95); border-right: 1px solid rgba(99,102,241,0.2); }
[data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px; padding: 1.5rem;
    text-align: center; backdrop-filter: blur(10px);
}
.metric-value { font-size: 2rem; font-weight: 700; color:
.metric-label { font-size: 0.85rem; color:
h1 { background: linear-gradient(90deg,
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 700; }
h2, h3 { color:
.stSelectbox label, .stSlider label { color:
</style>
    experiments = {}
    if not os.path.isdir(LOG_DIR):
        return experiments
    for name in os.listdir(LOG_DIR):
        log_path = os.path.join(LOG_DIR, name, "training_log.json")
        if os.path.isfile(log_path):
            experiments[name] = log_path
    return experiments


def load_log(path):
    with open(path) as f:
        return json.load(f)


def load_summary(exp_name):
    path = os.path.join(LOG_DIR, exp_name, "summary.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_tournament():
    path = os.path.join(RESULTS_DIR, "tournament", "tournament_results.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


with st.sidebar:
    st.markdown("# 🤖 MARL System")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏋️ Training Monitor", "🌍 Environment Viewer",
         "📊 Experiment Comparison", "🏆 Leaderboard"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.75rem;text-align:center'>"
        "Multi-Agent RL Training System<br>v1.0.0</div>",
        unsafe_allow_html=True,
    )

experiments = find_experiments()

if page == "🏋️ Training Monitor":
    st.markdown("# 🏋️ Training Monitor")

    if not experiments:
        st.info("No training runs found. Run `python train.py` to start training.")
    else:
        exp_name = st.selectbox("Select Experiment", list(experiments.keys()))
        log = load_log(experiments[exp_name])
        summary = load_summary(exp_name)

        if summary:
            cols = st.columns(4)
            metrics = [
                ("Episodes", str(summary.get("total_episodes", "—"))),
                ("Best Eval", f"{summary.get('best_eval_reward', 0):.2f}"),
                ("Final Reward", f"{summary.get('final_mean_reward', 0):.2f}"),
                ("Duration", f"{summary.get('elapsed_seconds', 0):.0f}s"),
            ]
            for col, (label, value) in zip(cols, metrics):
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{value}</div>"
                    f"<div class='metric-label'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                training_reward_curves(log, f"Rewards — {exp_name}"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                loss_curve(log, f"Loss — {exp_name}"),
                use_container_width=True,
            )

        with st.expander("📋 Raw Training Log (last 10 episodes)"):
            st.json(log[-10:])

elif page == "🌍 Environment Viewer":
    st.markdown("# 🌍 Environment Viewer")

    if not experiments:
        st.info("No training runs found.")
    else:
        exp_name = st.selectbox("Select Experiment", list(experiments.keys()),
                                key="env_viewer_exp")
        log = load_log(experiments[exp_name])

        env_type = None
        if "traffic" in exp_name.lower():
            env_type = "traffic"
        elif "trading" in exp_name.lower():
            env_type = "trading"
        elif "supply" in exp_name.lower():
            env_type = "supply_chain"

        st.markdown(f"**Detected environment:** `{env_type or 'unknown'}`")

        episode_idx = st.slider(
            "Episode", 0, len(log) - 1, len(log) - 1, key="env_ep"
        )
        episode = log[episode_idx]

        st.markdown(f"### Episode {episode.get('episode', episode_idx)}")
        st.markdown(
            f"Mean reward: **{episode.get('mean_reward', 0):.2f}** | "
            f"Steps: **{episode.get('steps', 0)}**"
        )

        if env_type == "traffic":
            sample_snapshot = {
                "step": episode.get("steps", 0),
                "queues": [[3, 5, 2, 4], [1, 7, 3, 2], [4, 2, 6, 1], [2, 3, 1, 5]],
            }
            st.plotly_chart(traffic_heatmap(sample_snapshot),
                           use_container_width=True)

        elif env_type == "trading":
            import numpy as np
            np.random.seed(42)
            prices = [100.0]
            for _ in range(200):
                prices.append(prices[-1] * np.exp(np.random.normal(0.0002, 0.015)))
            price_history = [[p, p*1.01, p*0.99] for p in prices]
            st.plotly_chart(
                trading_candlestick(price_history, 0, "Asset 0 Price"),
                use_container_width=True,
            )

        elif env_type == "supply_chain":
            sample_snapshot = {
                "step": episode.get("steps", 0),
                "inventory": [120, 85, 60, 30],
                "backorders": [0, 5, 15, 25],
            }
            st.plotly_chart(supply_chain_flow(sample_snapshot),
                           use_container_width=True)

elif page == "📊 Experiment Comparison":
    st.markdown("# 📊 Experiment Comparison")

    if len(experiments) < 2:
        st.info("Need at least 2 experiments to compare. "
                "Run training with different configs.")
    else:
        selected = st.multiselect(
            "Select experiments to compare",
            list(experiments.keys()),
            default=list(experiments.keys())[:3],
        )
        if selected:
            exp_logs = {name: load_log(experiments[name]) for name in selected}
            st.plotly_chart(
                experiment_comparison(exp_logs),
                use_container_width=True,
            )

            st.markdown("### Summary Table")
            rows = []
            for name in selected:
                s = load_summary(name)
                rows.append({
                    "Experiment": name,
                    "Episodes": s.get("total_episodes", "—"),
                    "Best Eval": f"{s.get('best_eval_reward', 0):.2f}",
                    "Final Reward": f"{s.get('final_mean_reward', 0):.2f}",
                    "Duration": f"{s.get('elapsed_seconds', 0):.0f}s",
                })
            st.table(rows)

elif page == "🏆 Leaderboard":
    st.markdown("# 🏆 Tournament Leaderboard")

    tournament_data = load_tournament()
    if tournament_data is None:
        st.info("No tournament results found. "
                "Run `python run_tournament.py` first.")
    else:
        elo = tournament_data.get("elo_ratings", {})
        st.plotly_chart(elo_leaderboard(elo), use_container_width=True)

        st.markdown("### Match History")
        history = tournament_data.get("match_history", [])
        if history:
            st.table(history[-20:])
        else:
            st.write("No matches recorded.")

