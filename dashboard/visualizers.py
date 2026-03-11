
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def training_reward_curves(
    training_log: List[Dict[str, Any]],
    title: str = "Training Rewards",
) -> go.Figure:
    episodes = [r["episode"] for r in training_log]
    mean_rewards = [r["mean_reward"] for r in training_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=mean_rewards,
        mode="lines", name="Mean Reward",
        line=dict(color="#6366f1", width=3),
    ))

    if training_log and "rewards" in training_log[0]:
        agent_ids = list(training_log[0]["rewards"].keys())
        colors = ["#f43f5e", "#10b981", "#f59e0b", "#3b82f6",
                  "#8b5cf6", "#ec4899", "#14b8a6", "#ef4444"]
        for i, aid in enumerate(agent_ids):
            agent_rewards = [r["rewards"].get(aid, 0) for r in training_log]
            fig.add_trace(go.Scatter(
                x=episodes, y=agent_rewards,
                mode="lines", name=f"Agent {aid}",
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dot"),
                opacity=0.7,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def loss_curve(
    training_log: List[Dict[str, Any]],
    title: str = "Training Loss",
) -> go.Figure:
    episodes = []
    losses = []
    for r in training_log:
        if "learn_metrics" in r:
            for aid, metrics in r["learn_metrics"].items():
                if isinstance(metrics, dict) and "loss" in metrics:
                    episodes.append(r["episode"])
                    losses.append(metrics["loss"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=losses,
        mode="lines", name="Loss",
        line=dict(color="#f43f5e", width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Loss",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def traffic_heatmap(snapshot: Dict[str, Any]) -> go.Figure:
    queues = np.array(snapshot.get("queues", []))
    if queues.ndim < 2:
        queues = queues.reshape(-1, 4)

    directions = ["North", "East", "South", "West"]
    agents = [f"Int {i}" for i in range(len(queues))]

    fig = go.Figure(data=go.Heatmap(
        z=queues,
        x=directions,
        y=agents,
        colorscale="YlOrRd",
        text=np.round(queues, 1),
        texttemplate="%{text}",
        colorbar=dict(title="Queue"),
    ))
    fig.update_layout(
        title=f"Traffic Queue Lengths — Step {snapshot.get('step', '?')}",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig


def trading_candlestick(
    price_history: List[List[float]],
    asset_idx: int = 0,
    title: str = "Asset Price",
) -> go.Figure:
    prices = [p[asset_idx] if isinstance(p, (list, np.ndarray)) else p
              for p in price_history]
    steps = list(range(len(prices)))

    window = 10
    ma = []
    for i in range(len(prices)):
        start = max(0, i - window + 1)
        ma.append(np.mean(prices[start:i+1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=prices,
        mode="lines", name="Price",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=ma,
        mode="lines", name=f"MA({window})",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Price",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def supply_chain_flow(snapshot: Dict[str, Any]) -> go.Figure:
    roles = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
    inventory = snapshot.get("inventory", [0]*4)
    backorders = snapshot.get("backorders", [0]*4)

    n = min(len(inventory), len(roles))
    roles = roles[:n]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=roles, y=inventory[:n],
        name="Inventory",
        marker_color="#10b981",
    ))
    fig.add_trace(go.Bar(
        x=roles, y=backorders[:n],
        name="Backorders",
        marker_color="#f43f5e",
    ))
    fig.update_layout(
        title=f"Supply Chain Status — Step {snapshot.get('step', '?')}",
        barmode="group",
        xaxis_title="Role",
        yaxis_title="Units",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def elo_leaderboard(elo_ratings: Dict[str, float]) -> go.Figure:
    sorted_agents = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    names = [a[0] for a in sorted_agents]
    ratings = [a[1] for a in sorted_agents]

    colors = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6"]

    fig = go.Figure(go.Bar(
        x=ratings, y=names,
        orientation="h",
        marker_color=colors[:len(names)],
        text=[f"{r:.0f}" for r in ratings],
        textposition="outside",
    ))
    fig.update_layout(
        title="ELO Leaderboard",
        xaxis_title="ELO Rating",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=100, r=60, t=60, b=50),
    )
    return fig


def experiment_comparison(
    experiments: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    fig = go.Figure()
    colors = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6",
              "#8b5cf6", "#ec4899", "#14b8a6"]

    for i, (name, log) in enumerate(experiments.items()):
        episodes = [r["episode"] for r in log]
        rewards = [r["mean_reward"] for r in log]
        fig.add_trace(go.Scatter(
            x=episodes, y=rewards,
            mode="lines", name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Experiment Comparison",
        xaxis_title="Episode",
        yaxis_title="Mean Reward",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig

