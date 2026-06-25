# AI-Assisted Air Traffic Management in the Northeast Corridor
### D.Eng. Praxis — George Washington University

**Author:** Nicholas [Last Name]  
**Advisor:** Dr. Nila Fridley  
**Praxis Director:** Dr. Amir Etemadi  
**Committee Member:** Dr. Timothy Blackburn  
**Defense Date:** July 23, 2026  

---

## Overview

This repository contains the simulation environment, training code, and evaluation scripts for a D.Eng. praxis investigating the application of deep reinforcement learning (DRL) to air traffic conflict management in the Northeast Corridor. A Proximal Policy Optimization (PPO) agent is trained within a custom Gymnasium environment (ATCEnv) to reduce aircraft separation conflicts and loss-of-separation events under varying traffic density and frequency conditions.

The system-of-systems (SoS) architecture is modeled in SysML v2 using Cameo Systems Modeler, framing the AI agent as an advisory component integrated within existing ATC infrastructure.

---

## Research Hypotheses

| Hypothesis | Description | Outcome |
|---|---|---|
| H1 | PPO agent reduces conflict alerts by ≥ 20% vs. baseline | **PASS** — 36.4% reduction |
| H2 (LoS) | PPO agent reduces loss-of-separation events vs. baseline | **PASS** — 36.4% reduction |
| H2 (TUC) | PPO agent reduces time-under-conflict vs. baseline | **FAIL** — not statistically significant |
| H3 | Traffic density × frequency interaction affects agent performance | **Not Supported** — interaction significant (F(4,261)=3.480, p=0.0086) but no individual conditions reached significance |

---

## Tech Stack

| Component | Technology |
|---|---|
| RL Algorithm | Proximal Policy Optimization (PPO) — Stable-Baselines3 |
| Environment | Custom Gymnasium environment (ATCEnv) |
| Observation Space | 75-dimensional continuous vector |
| Training | 1,000,000 timesteps |
| Evaluation | 30 Monte Carlo runs per experimental condition |
| Modeling | SysML v2 — Cameo Systems Modeler |
| Language | Python 3.x |
| IDE | PyCharm |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YOURUSERNAME/deng-praxis.git
cd deng-praxis

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
```
stable-baselines3
gymnasium
numpy
pandas
matplotlib
```

---

## Running the Code

**Train the PPO agent:**
```bash
python training/train.py
```

**Run Monte Carlo evaluation (30 runs per condition):**
```bash
python evaluation/evaluate.py
```

---

## Key Results

The PPO agent was evaluated across multiple traffic density and frequency conditions using 30 Monte Carlo runs per condition:

- **H1:** Conflict alert reduction of **36.4%** relative to baseline (p < 0.05) ✅
- **H2 (LoS):** Loss-of-separation reduction of **36.4%** relative to baseline (p < 0.05) ✅
- **H2 (TUC):** Time-under-conflict reduction did not reach statistical significance ❌
- **H3:** A significant density × frequency interaction was detected (F(4,261)=3.480, p=0.0086), but no individual experimental conditions reached significance — H3 not supported

---

## Academic Context

This praxis was submitted in partial fulfillment of the requirements for the **Doctor of Engineering (D.Eng.)** degree at the **George Washington University**, School of Engineering and Applied Science.

**Research domain:** AI-assisted decision support in safety-critical systems  
**Application domain:** Federal Aviation Administration (FAA) Northeast Corridor airspace  

---

## AI Use Disclosure

Code generation for this project was assisted by AI tools in accordance with GWU's AI use policy for D.Eng. candidates. All AI-generated code has been reviewed, tested, and attributed per institutional guidelines. Prose drafting was completed without AI assistance per policy requirements.

---

## License

This repository is private and intended for academic review purposes only. All rights reserved © 2026.
