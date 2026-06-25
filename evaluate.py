# -*- coding: utf-8 -*-
"""
=============================================================================
 evaluate.py
 Monte Carlo Evaluation Script — PPO ATC Conflict Management Agent
 Northeast Corridor (NEC) Airspace Simulation

 Nicholas D. Spataro, D.Eng. Candidate
 George Washington University, 2026

 Part of: AI-Assisted Air Traffic Management in the Northeast Corridor
 D.Eng. Praxis — Appendix B

 Description:
   Loads the trained PPO model and runs Monte Carlo evaluation across
   30 episodes per condition to reproduce Chapter 4 hypothesis results.
   All statistical outputs, figures, and summary tables are saved to
   results_ppo/.

 Requires (in results_ppo/):
   ppo_atc_model_final.zip      Trained PPO model (output of train.py)
   vecnormalize.pkl             VecNormalize statistics (output of train.py)
   eval_baseline_aircraft.csv   Fixed baseline states (created by train.py)

 Produces (in results_ppo/):
   h1_h2_results.csv            Raw Monte Carlo data (H1/H2)
   h1_h2_summary.txt            Table 4-1 and Table 4-2 numbers
   h3_results.csv               Raw H3 factorial data
   hypothesis_summary.txt       Full Chapter 4 results summary
   figure_4_1_traj.png          Figure 4-1: Trajectory comparison
   figure_4_2_boxplot.png       Figure 4-2: Conflict distribution boxplots
   figure_4_3_tornado.png       Figure 4-3: H3 sensitivity tornado plot

 Usage:
   python evaluate.py

 Dependencies:
   stable-baselines3, gymnasium, numpy, pandas, scipy, matplotlib
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import time, os, csv, warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from atc_env import (
    ATCEnv, MAX_AIRCRAFT, TRAINED_OBS, DENSITY_LEVELS, FREQ_LEVELS,
    haversine_dist, latlon_to_nm, get_vertical_min,
    ADV_NM, ADV_FT, SEPARATION_MIN, _DATA_CACHE,
    build_gradual_approach_baseline
)

os.makedirs("results_ppo", exist_ok=True)


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model():
    """
    Load the trained PPO model and VecNormalize statistics.
    Returns the model and the unwrapped raw ATCEnv instance.
    """
    print("=== LOADING MODEL ===")
    vec_env = make_vec_env(ATCEnv, n_envs=1)
    vec_env = VecNormalize.load("results_ppo/vecnormalize.pkl", vec_env)
    model   = PPO.load("results_ppo/ppo_atc_model_final", env=vec_env)
    raw_env = vec_env.envs[0].env.unwrapped

    # Load or build fixed baseline aircraft states
    _csv = "results_ppo/eval_baseline_aircraft.csv"
    if os.path.exists(_csv):
        raw_env.base_states = pd.read_csv(_csv)
        print(f"[FIXED DATASET] Loaded eval_baseline_aircraft.csv")
    else:
        raw_env.base_states = build_gradual_approach_baseline(MAX_AIRCRAFT, seed=42)
        raw_env.base_states.to_csv(_csv, index=False)
        print(f"[BASELINE SAVED] eval_baseline_aircraft.csv created")

    _DATA_CACHE.clear()
    _DATA_CACHE[MAX_AIRCRAFT] = raw_env.base_states.copy()
    print(f"Baseline: {len(raw_env.base_states)} aircraft ready.\n")
    return model, raw_env


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================
def _pct(b, a):
    """Percentage reduction from baseline b to AI a."""
    return 0.0 if b == 0 else (b - a) / b * 100.0

def _ttest(b, a):
    """Paired t-test between baseline and AI arrays."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, p = stats.ttest_rel(np.array(b, float), np.array(a, float))
    return float(t), float(p)

def _cohens_d(b, a):
    """Cohen's d effect size for paired comparison."""
    b, a = np.array(b, float), np.array(a, float)
    sp = np.sqrt(
        ((len(b) - 1) * b.std(ddof=1) ** 2 + (len(a) - 1) * a.std(ddof=1) ** 2)
        / (len(b) + len(a) - 2)
    )
    return float((b.mean() - a.mean()) / sp) if sp > 0 else 0.0

def _count_tuc(tuc, states):
    """Increment time-under-conflict counter if any aircraft pair is within advisory zone."""
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            s, t = states[i], states[j]
            if (haversine_dist(s["lat"], s["lon"], t["lat"], t["lon"]) < ADV_NM
                    and abs(s["alt_ft"] - t["alt_ft"]) < ADV_FT):
                return tuc + 1
    return tuc


# =============================================================================
# H1 + H2 EVALUATION
# =============================================================================
def run_h1_h2(model, raw_env, num_runs=100, steps=1800):
    """
    Run Monte Carlo evaluation for H1 (conflict alert reduction) and
    H2 (TUC workload reduction and hard LoS reduction).

    Parameters
    ----------
    model    : PPO model loaded from results_ppo/
    raw_env  : Unwrapped ATCEnv instance
    num_runs : Number of Monte Carlo episodes (default: 100)
    steps    : Simulation steps per episode (default: 1800 = 30 min)

    Returns
    -------
    b_res, a_res : lists of per-episode dicts with conflict_count,
                   tuc_seconds, hard_los_count
    """
    print("\n" + "=" * 65)
    print(f"H1 + H2  |  {num_runs} Monte Carlo runs  |  {MAX_AIRCRAFT} aircraft")
    print("=" * 65)
    t0 = time.time()
    b_res = []
    a_res = []

    for run in range(num_runs):
        if run % 5 == 0:
            print(f"  Run {run + 1}/{num_runs}")

        # --- BASELINE (no AI intervention, random heading drift) ---
        current = raw_env.base_states.copy()
        current['heading'] += np.random.normal(0, 40, MAX_AIRCRAFT)
        bc = bt = bl = 0
        for _ in range(steps):
            current = raw_env._update_positions(current)
            current['heading'] += np.random.normal(0, 5, MAX_AIRCRAFT)
            pos_nm = latlon_to_nm(current[['lat', 'lon']].values)
            alts   = current['alt'].values
            bc += raw_env._compute_conflicts(pos_nm, alts)
            bl += raw_env._compute_hard_los(pos_nm, alts)
            bt  = _count_tuc(bt, [
                {"lat": float(current.iloc[i]['lat']),
                 "lon": float(current.iloc[i]['lon']),
                 "alt_ft": float(current.iloc[i]['alt'])}
                for i in range(MAX_AIRCRAFT)
            ])
        b_res.append({"conflict_count": bc, "tuc_seconds": bt, "hard_los_count": bl})

        # --- AI AGENT ---
        obs = raw_env.reset()[0]
        ac = at = al = 0
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = raw_env.step(action)
            ac += info.get("n_conflicts", 0)
            al += info.get("n_hard_los", 0)
            if info.get("aircraft_states"):
                at = _count_tuc(at, info["aircraft_states"])
            if term or trunc:
                break
        a_res.append({"conflict_count": ac, "tuc_seconds": at, "hard_los_count": al})

    print(f"  Completed in {time.time() - t0:.0f}s")
    return b_res, a_res


def analyze_and_save(b_res, a_res):
    """
    Compute statistics and save H1/H2 results to disk.
    Returns a dict of all computed values for figure generation.
    """
    ba = np.array([r["conflict_count"] for r in b_res], float)
    aa = np.array([r["conflict_count"] for r in a_res], float)
    bt = np.array([r["tuc_seconds"]    for r in b_res], float)
    at = np.array([r["tuc_seconds"]    for r in a_res], float)
    bl = np.array([r["hard_los_count"] for r in b_res], float)
    al = np.array([r["hard_los_count"] for r in a_res], float)

    # H1: Conflict alert reduction
    h1_red  = _pct(ba.mean(), aa.mean())
    h1_t, h1_p = _ttest(ba, aa)
    h1_d    = _cohens_d(ba, aa)
    h1_pass = h1_red >= 15.0 and h1_p < 0.05

    # H2a: TUC reduction
    tr, tp  = _pct(bt.mean(), at.mean()), _ttest(bt, at)[1]
    tuc_pass = tr >= 25.0 and tp < 0.05

    # H2b: Hard LoS reduction
    lr      = _pct(bl.mean(), al.mean())
    lt, lp  = _ttest(bl, al)
    ld      = _cohens_d(bl, al)
    los_pass = lr >= 20.0 and lp < 0.05

    # Print results
    print("\n" + "=" * 65)
    print(f"  H1: Conflict Alert Reduction (target >= 15%)")
    print(f"  Baseline : {ba.mean():.1f}  SD={ba.std():.1f}  min={ba.min():.0f}  max={ba.max():.0f}")
    print(f"  AI       : {aa.mean():.1f}  SD={aa.std():.1f}  min={aa.min():.0f}  max={aa.max():.0f}")
    print(f"  Reduction: {h1_red:.1f}%  t={h1_t:.3f}  p={h1_p:.4f}  Cohen's d={h1_d:.2f}")
    print(f"  Result   : {'[PASS] H1 SUPPORTED' if h1_pass else '[FAIL]'}")

    print(f"\n  H2: TUC Workload (target >= 25%)")
    print(f"  Baseline: {bt.mean():.1f}s  AI: {at.mean():.1f}s  Red: {tr:.1f}%  p={tp:.4f}")
    print(f"  Result   : {'[PASS]' if tuc_pass else '[FAIL]'}")

    print(f"\n  H2: Hard LoS (target >= 20%)")
    print(f"  Baseline: {bl.mean():.2f}  AI: {al.mean():.2f}  Red: {lr:.1f}%  p={lp:.4f}  d={ld:.2f}")
    print(f"  Result   : {'[PASS]' if los_pass else '[FAIL]'}")
    print("=" * 65)

    # Save raw CSV
    with open("results_ppo/h1_h2_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "condition", "conflict_count", "tuc_seconds", "hard_los_count"])
        for cond, res in [("baseline", b_res), ("ai", a_res)]:
            for i, r in enumerate(res):
                w.writerow([i + 1, cond, r["conflict_count"], r["tuc_seconds"], r["hard_los_count"]])

    # Save summary text
    lines = [
        "=" * 65, "H1/H2 RESULTS  (100 Monte-Carlo runs)", "=" * 65, "",
        "TABLE 4-1: H1 Conflict Alert Reduction", "",
        f"  Baseline: {ba.mean():.1f} (SD={ba.std():.1f})  min={ba.min():.0f}  max={ba.max():.0f}",
        f"  AI:       {aa.mean():.1f} (SD={aa.std():.1f})  min={aa.min():.0f}  max={aa.max():.0f}",
        f"  Reduction: {h1_red:.1f}%",
        f"  t={h1_t:.3f}  p={h1_p:.4f}  Cohen's d={h1_d:.2f}",
        f"  H1: {'[PASS] SUPPORTED' if h1_pass else '[FAIL] NOT SUPPORTED'}  (target >= 15%)",
        "", "TABLE 4-2: H2 Safety Metrics", "",
        f"  TUC: Baseline={bt.mean():.1f}s  AI={at.mean():.1f}s  Red={tr:.1f}%  p={tp:.4f}  {'[PASS]' if tuc_pass else '[FAIL]'}",
        f"  LoS: Baseline={bl.mean():.2f}  AI={al.mean():.2f}  Red={lr:.1f}%  p={lp:.4f}  d={ld:.2f}  {'[PASS]' if los_pass else '[FAIL]'}",
        f"  H2 OVERALL: {'[PASS] SUPPORTED' if (tuc_pass and los_pass) else '[FAIL] NOT SUPPORTED'}",
        "", "=" * 65,
    ]
    with open("results_ppo/h1_h2_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Saved: h1_h2_results.csv + h1_h2_summary.txt")

    return {
        "ba": ba, "aa": aa, "bt": bt, "at": at, "bl": bl, "al": al,
        "h1_red": h1_red, "h1_t": h1_t, "h1_p": h1_p, "h1_d": h1_d, "h1_pass": h1_pass,
        "tr": tr, "tp": tp, "tuc_pass": tuc_pass,
        "lr": lr, "lp": lp, "ld": ld, "los_pass": los_pass,
    }


# =============================================================================
# FIGURE 4-1: TRAJECTORY COMPARISON
# =============================================================================
def plot_figure_4_1(model, raw_env):
    """Generate Figure 4-1: Baseline vs AI trajectory comparison plot."""
    print("\n  Generating Figure 4-1 (trajectories)...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, MAX_AIRCRAFT))

    for ax_idx, (use_ai, title) in enumerate([
        (False, "Baseline (No AI Intervention)"),
        (True,  "PPO AI Agent Active")
    ]):
        ax = axes[ax_idx]
        current = raw_env.base_states.copy()
        current['heading'] += np.random.normal(0, 40, MAX_AIRCRAFT)
        traj = {i: {"lats": [], "lons": []} for i in range(MAX_AIRCRAFT)}
        total_conf = 0

        if use_ai:
            obs = raw_env.reset()[0]
            for _ in range(1800):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, _, info = raw_env.step(action)
                total_conf += info["n_conflicts"]
                for i in range(MAX_AIRCRAFT):
                    traj[i]["lats"].append(raw_env.current.iloc[i]['lat'])
                    traj[i]["lons"].append(raw_env.current.iloc[i]['lon'])
                if term:
                    break
        else:
            for _ in range(1800):
                current = raw_env._update_positions(current)
                current['heading'] += np.random.normal(0, 5, MAX_AIRCRAFT)
                pos_nm = latlon_to_nm(current[['lat', 'lon']].values)
                total_conf += raw_env._compute_conflicts(pos_nm, current['alt'].values)
                for i in range(MAX_AIRCRAFT):
                    traj[i]["lats"].append(float(current.iloc[i]['lat']))
                    traj[i]["lons"].append(float(current.iloc[i]['lon']))

        for i in range(MAX_AIRCRAFT):
            lats = traj[i]["lats"]
            lons = traj[i]["lons"]
            if not lats:
                continue
            ax.plot(lons[::30], lats[::30], color=colors[i], linewidth=1.8, alpha=0.85)
            ax.plot(lons[0],  lats[0],  'o', color=colors[i], markersize=7, zorder=5)
            ax.plot(lons[-1], lats[-1], '^', color=colors[i], markersize=7, zorder=5)

        ax.set_xlabel("Longitude (°W)", fontsize=11)
        ax.set_ylabel("Latitude (°N)",  fontsize=11)
        ax.set_title(f"{title}\nTotal Conflict Steps: {total_conf}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-78, -71)
        ax.set_ylim(38, 43)

        rect = Rectangle((-78, 38), 7, 5, linewidth=1.5, edgecolor='red',
                         facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(rect)
        ax.text(-77.8, 42.7, "NEC Airspace", fontsize=8, color='red', alpha=0.7)

    legend_elements = [
        mpatches.Patch(color='gray', label='Aircraft trajectory'),
        plt.Line2D([0], [0], marker='o', color='gray', markersize=6,
                   label='Start position', linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='gray', markersize=6,
                   label='End position',  linestyle='None'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.suptitle(
        "Figure 4-1: NEC En-Route Aircraft Trajectories\n"
        "Baseline (No AI) vs. PPO AI Agent  —  Nicholas D. Spataro, D.Eng., 2026",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("results_ppo/figure_4_1_traj.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results_ppo/figure_4_1_traj.png")


# =============================================================================
# FIGURE 4-2: CONFLICT DISTRIBUTION BOXPLOTS
# =============================================================================
def plot_figure_4_2(stats_dict):
    """Generate Figure 4-2: Conflict and LoS distribution boxplots."""
    print("  Generating Figure 4-2 (boxplots)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data_pair, ylabel, title_str, red_key, p_key, d_key in [
        (axes[0],
         [stats_dict["ba"], stats_dict["aa"]],
         "Conflict Alert Count per Episode",
         f"H1: Conflict Alert Distribution\nReduction: {stats_dict['h1_red']:.1f}%  p={stats_dict['h1_p']:.4f}  d={stats_dict['h1_d']:.2f}",
         "h1_red", "h1_p", "h1_d"),
        (axes[1],
         [stats_dict["bl"], stats_dict["al"]],
         "Hard LoS Breach Count per Episode",
         f"H2: Hard Loss-of-Separation Distribution\nReduction: {stats_dict['lr']:.1f}%  p={stats_dict['lp']:.4f}  d={stats_dict['ld']:.2f}",
         "lr", "lp", "ld"),
    ]:
        bp = ax.boxplot(data_pair, labels=["Baseline\n(No AI)", "PPO AI\nAgent"],
                        patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#ff7f7f')
        bp['boxes'][1].set_facecolor('#7fbfff')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_str, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        y_max = max(data_pair[0].max(), data_pair[1].max()) * 1.05
        ax.plot([1, 1, 2, 2], [y_max * 0.95, y_max, y_max, y_max * 0.95], 'k-', linewidth=1)
        p_val = stats_dict[p_key]
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        ax.text(1.5, y_max * 1.01, sig, ha='center', fontsize=13)

    plt.suptitle(
        "Figure 4-2: Conflict and Loss-of-Separation Distributions\n"
        "100 Monte-Carlo Runs  —  Nicholas D. Spataro, D.Eng., 2026",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("results_ppo/figure_4_2_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results_ppo/figure_4_2_boxplot.png")


# =============================================================================
# H3 EVALUATION — 3×3 FACTORIAL
# =============================================================================
def _h3_episode(model, n_aircraft, interval, use_ai, steps=1800):
    """Run a single H3 episode for one density/frequency condition."""
    env = ATCEnv(n_aircraft=n_aircraft, ai_step_interval=interval)
    obs, _ = env.reset()
    conf = los = 0
    for _ in range(steps):
        if use_ai:
            padded = np.zeros(TRAINED_OBS, dtype=np.float32)
            padded[:min(len(obs), TRAINED_OBS)] = obs[:TRAINED_OBS]
            full_action, _ = model.predict(padded, deterministic=True)
            action = np.zeros(n_aircraft * 3, dtype=np.float32)
            copy_len = min(len(full_action), n_aircraft * 3)
            action[:copy_len] = full_action[:copy_len]
        else:
            action = np.zeros(n_aircraft * 3, dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        conf += info.get("n_conflicts", 0)
        los  += info.get("n_hard_los", 0)
        if term or trunc:
            break
    env.close()
    return conf + los


def run_h3(model, num_runs=100, steps=1800):
    """
    Run 3×3 factorial experiment (H3): density × update frequency.
    Density levels: LOW (10ac), BASE (15ac), HIGH (20ac)
    Frequency levels: FAST (1s), MID (5s), SLOW (10s)
    100 Monte Carlo runs per cell.
    """
    print("\n" + "=" * 65)
    print(f"H3  |  3×3 factorial  |  {num_runs} runs/cell")
    print("=" * 65)
    grid = {}
    cell = 0

    for d_label, n_ac in DENSITY_LEVELS.items():
        for f_label, interval in FREQ_LEVELS.items():
            cell += 1
            print(f"  Cell {cell}/9: {d_label}({n_ac}ac) {f_label}({interval}s)")
            bs = [_h3_episode(model, n_ac, interval, False, steps) for _ in range(num_runs)]
            ai = [_h3_episode(model, n_ac, interval, True,  steps) for _ in range(num_runs)]
            red = _pct(np.mean(bs), np.mean(ai))
            _, p = _ttest(bs, ai)
            grid[(d_label, f_label)] = {"baseline": bs, "ai": ai, "red": red, "p": p}
            print(f"  → Base:{np.mean(bs):.1f}  AI:{np.mean(ai):.1f}  "
                  f"Red:{red:.1f}%  p={p:.4f}  "
                  f"{'[PASS]' if red >= 10 and p < 0.05 else '[FAIL]'}")

            # Autosave after each cell
            with open("results_ppo/h3_results.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["density", "freq", "condition", "run", "risk_score"])
                for d in DENSITY_LEVELS:
                    for freq in FREQ_LEVELS:
                        if (d, freq) not in grid:
                            continue
                        for cond in ["baseline", "ai"]:
                            for i, s in enumerate(grid[(d, freq)][cond]):
                                w.writerow([d, freq, cond, i + 1, s])
    return grid


# =============================================================================
# FIGURE 4-3: TORNADO PLOT (H3 SENSITIVITY)
# =============================================================================
def plot_figure_4_3(grid):
    """Generate Figure 4-3: Tornado plot showing H3 sensitivity analysis."""
    print("  Generating Figure 4-3 (tornado plot)...")
    density_reds  = [grid[(d, "FAST")]["red"] for d in DENSITY_LEVELS]
    freq_reds     = [grid[("BASE", f)]["red"] for f in FREQ_LEVELS]
    density_range = max(density_reds) - min(density_reds)
    freq_range    = max(freq_reds)    - min(freq_reds)

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [
        "Traffic Density\n(LOW / BASE / HIGH)",
        "AI Update Frequency\n(FAST 1s / MID 5s / SLOW 10s)"
    ]
    values     = [density_range, freq_range]
    colors_bar = ["#2196F3", "#FF9800"]
    bars = ax.barh(labels, values, color=colors_bar, height=0.4,
                   edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"±{val:.1f} pp", va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel("Range of Risk Score Reduction (percentage points)", fontsize=11)
    ax.set_title(
        "Figure 4-3: Tornado Plot — H3 Sensitivity Analysis\n"
        "Sensitivity of Risk Reduction to Experimental Factors",
        fontsize=12
    )
    ax.set_xlim(0, max(values) * 1.6 + 1)
    ax.grid(axis='x', alpha=0.35)
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.text(10.2, 1.45, "H3 threshold (10%)", color='green', fontsize=8, alpha=0.8)

    plt.tight_layout()
    plt.savefig("results_ppo/figure_4_3_tornado.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results_ppo/figure_4_3_tornado.png")
    return {"density_range": density_range, "freq_range": freq_range}


# =============================================================================
# FINAL SUMMARY
# =============================================================================
def save_final_summary(s, h3_grid, tornado):
    """Save full hypothesis summary to results_ppo/hypothesis_summary.txt."""
    h3_pass = all(v["red"] >= 10.0 and v["p"] < 0.05 for v in h3_grid.values())
    lines = [
        "=" * 70, "SPATARO PRAXIS — FULL HYPOTHESIS RESULTS",
        "Nicholas D. Spataro, D.Eng., GWU 2026", "=" * 70, "",
        "TABLE 4-1: H1 — Conflict Alert Reduction (30 Monte-Carlo Runs)",
        f"  Baseline: {s['ba'].mean():.1f} (SD={s['ba'].std():.1f})  min={s['ba'].min():.0f}  max={s['ba'].max():.0f}",
        f"  AI:       {s['aa'].mean():.1f} (SD={s['aa'].std():.1f})  min={s['aa'].min():.0f}  max={s['aa'].max():.0f}",
        f"  Reduction: {s['h1_red']:.1f}%  t={s['h1_t']:.3f}  p={s['h1_p']:.4f}  Cohen's d={s['h1_d']:.2f}",
        f"  H1: {'[PASS] SUPPORTED' if s['h1_pass'] else '[FAIL]'}  (target >= 15%)",
        "", "TABLE 4-2: H2 — Safety Metrics",
        f"  TUC: Base={s['bt'].mean():.1f}s  AI={s['at'].mean():.1f}s  Red={s['tr']:.1f}%  p={s['tp']:.4f}  {'[PASS]' if s['tuc_pass'] else '[FAIL]'}",
        f"  LoS: Base={s['bl'].mean():.2f}  AI={s['al'].mean():.2f}  Red={s['lr']:.1f}%  p={s['lp']:.4f}  d={s['ld']:.2f}  {'[PASS]' if s['los_pass'] else '[FAIL]'}",
        "", "TABLE 4-3: H3 — 3×3 Factorial Sensitivity",
        "  Density\\Freq    FAST(1s)    MID(5s)    SLOW(10s)",
    ]
    for d in DENSITY_LEVELS:
        row = f"  {d + ' (' + str(DENSITY_LEVELS[d]) + 'ac)':<18}"
        for f in FREQ_LEVELS:
            v = h3_grid[(d, f)]
            row += f"  {v['red']:>8.1f}%{'*' if v['red'] >= 10 and v['p'] < 0.05 else ' '}"
        lines.append(row)
    lines += [
        "  (* p<0.05, target >= 10%)",
        f"  Tornado: Density ±{tornado['density_range']:.1f}pp  |  Frequency ±{tornado['freq_range']:.1f}pp",
        f"  H3: {'[PASS] SUPPORTED' if h3_pass else '[FAIL] NOT SUPPORTED'}",
        "", "=" * 70,
        "FIGURES SAVED:",
        "  figure_4_1_traj.png    — Chapter 4 Figure 4-1",
        "  figure_4_2_boxplot.png — Chapter 4 Figure 4-2",
        "  figure_4_3_tornado.png — Chapter 4 Figure 4-3",
        "=" * 70,
    ]
    with open("results_ppo/hypothesis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Saved: results_ppo/hypothesis_summary.txt")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n=== STARTING FULL EVALUATION ===")
    t_start = time.time()

    model, raw_env = load_model()

    # H1 + H2
    b_res, a_res = run_h1_h2(model, raw_env, num_runs=100, steps=1800)
    s = analyze_and_save(b_res, a_res)

    # Figures 4-1 and 4-2
    print("\n-- Generating figures --")
    plot_figure_4_1(model, raw_env)
    plot_figure_4_2(s)

    # H3 factorial
    h3_grid = run_h3(model, num_runs=100, steps=1800)
    tornado  = plot_figure_4_3(h3_grid)
    save_final_summary(s, h3_grid, tornado)

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"  H1 (>= 15%):     {'[PASS]' if s['h1_pass'] else '[FAIL]'}  {s['h1_red']:.1f}%  p={s['h1_p']:.4f}")
    print(f"  H2 TUC (>= 25%): {'[PASS]' if s['tuc_pass'] else '[FAIL]'}  {s['tr']:.1f}%  p={s['tp']:.4f}")
    print(f"  H2 LoS (>= 20%): {'[PASS]' if s['los_pass'] else '[FAIL]'}  {s['lr']:.1f}%  p={s['lp']:.4f}")
    print(f"\n  Open results_ppo/hypothesis_summary.txt for full Chapter 4 numbers.")
    print(f"  Total runtime: {time.time() - t_start:.0f}s")