# -*- coding: utf-8 -*-
"""
=============================================================================
 atc_env.py
 ATCEnv — Custom Gymnasium Environment for ATC Conflict Management
 Northeast Corridor (NEC) Airspace Simulation

 Nicholas D. Spataro, D.Eng. Candidate
 George Washington University, 2026

 Part of: AI-Assisted Air Traffic Management in the Northeast Corridor
 D.Eng. Praxis — Appendix B

 Description:
   A custom Gymnasium environment simulating en-route air traffic in the
   Northeast Corridor (NEC). Used to train and evaluate a Proximal Policy
   Optimization (PPO) deep reinforcement learning agent for conflict
   detection and resolution.

 Observation Space:
   75-dimensional continuous vector (15 aircraft × 5 state variables:
   lat, lon, alt, velocity, heading — normalized)

 Action Space:
   45-dimensional continuous Box (15 aircraft × 3 control inputs:
   heading delta, altitude delta, acceleration delta)

 Dependencies:
   numpy, pandas, scipy, gymnasium, stable-baselines3
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from gymnasium import spaces, Env

# =============================================================================
# CONSTANTS
# =============================================================================
R_EARTH        = 3440        # Earth radius in nautical miles
MAX_AIRCRAFT   = 15          # Maximum number of aircraft in simulation
SEPARATION_MIN = 5.0         # Minimum horizontal separation (NM)
FL290          = 29000.0     # Flight Level 290 altitude threshold (ft)
HARD_NM        = 5.0         # Hard loss-of-separation horizontal threshold (NM)
HARD_LOW       = 1000.0      # Hard LoS vertical threshold below FL290 (ft)
HARD_HIGH      = 2000.0      # Hard LoS vertical threshold at/above FL290 (ft)
ADV_NM         = 8.0         # Advisory conflict horizontal threshold (NM)
ADV_FT         = 5000.0      # Advisory conflict vertical threshold (ft)
TRAINED_OBS    = MAX_AIRCRAFT * 5   # 75-dimensional observation space

# Northeast Corridor waypoints (lat, lon, airport code)
WAYPOINTS = [
    (40.639, -73.778, "JFK"),
    (40.692, -74.168, "EWR"),
    (39.872, -75.241, "PHL"),
    (38.852, -77.038, "DCA"),
    (39.175, -76.668, "BWI"),
    (42.364, -71.005, "BOS"),
    (41.939, -87.907, "ORD"),
    (33.942, -118.408, "LAX"),
]

# Aircraft performance envelopes
PERFORMANCE = {
    'jet':       {'max_climb': 50.0, 'max_turn': 3.0, 'max_accel': 1.0},
    'turboprop': {'max_climb': 25.0, 'max_turn': 3.0, 'max_accel': 0.5},
}

# H3 experimental factor levels
DENSITY_LEVELS = {"LOW": 10, "BASE": 15, "HIGH": 20}
FREQ_LEVELS    = {"FAST": 1, "MID": 5, "SLOW": 10}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two lat/lon points in nautical miles.
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R_EARTH * 60 * 2 * np.arcsin(np.sqrt(a))


def get_vertical_min(alt):
    """
    Return vertical separation minimum based on altitude (RVSM rules).
    1000 ft below FL290, 2000 ft at or above FL290.
    """
    return HARD_LOW if alt < FL290 else HARD_HIGH


def latlon_to_nm(pos):
    """
    Convert lat/lon in degrees to approximate NM-scaled Cartesian coordinates.
    Used for pairwise distance computation via scipy.
    """
    return np.deg2rad(pos) * R_EARTH


# =============================================================================
# DATA CACHE
# =============================================================================
_DATA_CACHE = {}


# =============================================================================
# ATCEnv — CUSTOM GYMNASIUM ENVIRONMENT
# =============================================================================
class ATCEnv(Env):
    """
    Custom Gymnasium environment simulating NEC en-route air traffic for
    deep reinforcement learning-based conflict management.

    Parameters
    ----------
    n_aircraft : int
        Number of aircraft in the simulation (default: MAX_AIRCRAFT = 15).
    ai_step_interval : int
        Number of simulation steps between AI control updates (used in H3
        factorial experiment to vary update frequency).

    Observation Space
    -----------------
    Box of shape (n_aircraft * 5,) — normalized lat, lon, alt, velocity,
    heading for each aircraft. Shape is always (75,) for the trained model.

    Action Space
    ------------
    Box of shape (n_aircraft * 3,) in [-1, 1] — scaled to heading delta
    (±8°), altitude delta (±300 ft), and acceleration delta (±15 kts).
    """

    def __init__(self, n_aircraft=MAX_AIRCRAFT, ai_step_interval=1):
        super().__init__()
        self.n_aircraft = n_aircraft
        self.ai_step_interval = ai_step_interval
        self._step_count_h3 = 0
        self._last_action = None
        self._prev_min_sep = None

        # Action space: heading, altitude, acceleration deltas per aircraft
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(n_aircraft * 3,), dtype=np.float32
        )

        # Observation space: 5 normalized state variables per aircraft
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(n_aircraft * 5,), dtype=np.float32
        )

        self.base_states = self._fetch_data()
        self.reset()

    def _fetch_data(self):
        """
        Generate or retrieve cached initial aircraft states.
        Aircraft are placed randomly near the NEC center (40.7°N, 74.0°W).
        """
        if self.n_aircraft in _DATA_CACHE:
            return _DATA_CACHE[self.n_aircraft].copy()

        data = []
        for i in range(self.n_aircraft):
            lat = 40.7 + np.random.normal(0, 0.65)
            lon = -74.0 + np.random.normal(0, 0.65)
            data.append([
                i, "UAL{}".format(i + 100), "JFK", "BOS", "jet", 0,
                lat, lon, 35000.0, 480.0, 90.0, 42.364, -71.005
            ])

        df = pd.DataFrame(data, columns=[
            "aircraft_id", "callsign", "dep_airport", "dest_airport",
            "ac_type", "time", "lat", "lon", "alt", "velocity",
            "heading", "dest_lat", "dest_lon"
        ])
        _DATA_CACHE[self.n_aircraft] = df.copy()
        return df

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state with randomized headings.

        Returns
        -------
        obs : np.ndarray
            Initial 75-dimensional normalized observation vector.
        info : dict
            Empty info dict (Gymnasium API compliance).
        """
        self.current = self.base_states.copy()
        self.current['heading'] += np.random.normal(0, 40, self.n_aircraft)
        self.step_count = 0
        self._step_count_h3 = 0
        self._last_action = None
        self._prev_min_sep = None
        self.prev_dist = np.array([
            haversine_dist(r['lat'], r['lon'], r['dest_lat'], r['dest_lon'])
            for _, r in self.current.iterrows()
        ])
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Build the normalized 75-dimensional observation vector.

        Normalization:
          lat:      (lat - 40.5) / 3.0
          lon:      (lon + 74.5) / 4.0
          alt:      (alt - 33000) / 10000
          velocity: (vel - 450) / 150
          heading:  heading / 180
        """
        raw = self.current[['lat', 'lon', 'alt', 'velocity', 'heading']].values.astype(np.float32)
        norm = raw.copy()
        norm[:, 0] = (raw[:, 0] - 40.5) / 3.0
        norm[:, 1] = (raw[:, 1] + 74.5) / 4.0
        norm[:, 2] = (raw[:, 2] - 33000) / 10000
        norm[:, 3] = (raw[:, 3] - 450) / 150
        norm[:, 4] = raw[:, 4] / 180
        return norm.flatten()

    def step(self, action):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        action : np.ndarray
            Agent action vector of shape (n_aircraft * 3,) in [-1, 1].
            Scaled internally to heading delta (±8°), altitude delta (±300 ft),
            acceleration delta (±15 kts), clipped to aircraft performance limits.

        Returns
        -------
        obs : np.ndarray
            Updated 75-dimensional observation.
        reward : float
            Step reward from _compute_reward().
        terminated : bool
            True after 1800 steps (30-minute episode).
        truncated : bool
            Always False.
        info : dict
            n_conflicts, n_hard_los, aircraft_states.
        """
        self._step_count_h3 += 1

        # Apply action at specified frequency (H3 update interval)
        if self._step_count_h3 % self.ai_step_interval == 0 or self._last_action is None:
            self._last_action = action.copy()
        effective = self._last_action

        # Scale and clip actions to performance limits
        av = (effective.reshape(self.n_aircraft, 3) * np.array([8, 300, 15])).astype(np.float32)
        for i in range(self.n_aircraft):
            perf = PERFORMANCE[self.current.iloc[i]['ac_type']]
            av[i, 0] = np.clip(av[i, 0], -perf['max_turn'],  perf['max_turn'])
            av[i, 1] = np.clip(av[i, 1], -perf['max_climb'], perf['max_climb'])
            av[i, 2] = np.clip(av[i, 2], -perf['max_accel'], perf['max_accel'])

        # Update positions
        self.current = self._update_positions(self.current)
        pos_nm = latlon_to_nm(self.current[['lat', 'lon']].values)
        alts = self.current['alt'].values

        # Progress toward destination
        current_dist = np.array([
            haversine_dist(r['lat'], r['lon'], r['dest_lat'], r['dest_lon'])
            for _, r in self.current.iterrows()
        ])

        # Compute reward and metrics
        reward = self._compute_reward(pos_nm, alts, self.prev_dist, self.current, av.flatten())
        self.prev_dist = current_dist
        self.step_count += 1
        terminated = self.step_count >= 1800

        n_conf = self._compute_conflicts(pos_nm, alts)
        n_hard = self._compute_hard_los(pos_nm, alts)

        ac_states = [
            {
                "lat":    float(self.current.iloc[i]['lat']),
                "lon":    float(self.current.iloc[i]['lon']),
                "alt_ft": float(self.current.iloc[i]['alt'])
            }
            for i in range(self.n_aircraft)
        ]

        return self._get_obs(), reward, terminated, False, {
            "n_conflicts":    n_conf,
            "n_hard_los":     n_hard,
            "aircraft_states": ac_states
        }

    def _update_positions(self, df):
        """
        Propagate aircraft positions forward by one second using heading and velocity.
        """
        df = df.copy()
        hr = np.radians(df['heading'])
        dt = 1 / 3600.0          # 1 second in hours
        d  = df['velocity'] * dt  # distance traveled (NM)
        df['lat'] += d * np.sin(hr) / 60.0
        df['lon'] += d * np.cos(hr) / 60.0
        df['time'] += 1
        return df

    def _compute_reward(self, positions, alts, prev_dist, df, actions):
        """
        Compute step reward.

        Reward components:
          - Conflict penalty:   -280 per advisory conflict pair
          - Severe LoS penalty: -480 per hard LoS pair
          - Proximity shaping:  -0.45 × (horizontal + vertical proximity sum)
          - Progress bonus:     +0.13 × distance-to-destination reduction (NM)
          - Conflict-free bonus: +2200 if zero conflicts this step
          - Boundary penalty:   -150 per degree outside NEC bounds
          - Action regularizer: -0.006 × L1 norm of actions
        """
        hd = squareform(pdist(positions))
        vd = np.abs(alts[:, None] - alts[None, :])
        np.fill_diagonal(hd, 100.0)
        np.fill_diagonal(vd, 20000.0)

        vm = get_vertical_min(alts.mean())

        conflicts = np.sum((hd < SEPARATION_MIN) & (vd < vm)) // 2
        severe    = np.sum((hd < 3.0)            & (vd < vm / 2)) // 2

        h_prox = np.clip((12.0 - hd) / 12.0, 0, None).sum()
        v_prox = np.clip((vm * 2.5 - vd) / (vm * 2.5), 0, None).sum()

        curr_dist = np.array([
            haversine_dist(r['lat'], r['lon'], r['dest_lat'], r['dest_lon'])
            for _, r in df.iterrows()
        ])
        progress = np.clip(prev_dist - curr_dist, -20, 20).sum()

        lats = df['lat'].values
        lons = df['lon'].values
        bpen = (
            np.sum(np.maximum(0, lats - 43) + np.maximum(0, 38 - lats)) +
            np.sum(np.maximum(0, lons + 71) + np.maximum(0, -78 - lons))
        ) * 150.0

        return float(
            -280.0 * conflicts
            - 480.0 * severe
            - 0.45  * (h_prox + v_prox)
            + progress * 0.13
            + (2200.0 if conflicts == 0 else 0.0)
            - bpen
            - np.sum(np.abs(actions)) * 0.006
        )

    def _compute_conflicts(self, positions, alts):
        """
        Count advisory-level conflict pairs (horizontal < 5 NM and vertical
        below RVSM minimum).
        """
        if len(positions) < 2:
            return 0
        hd = squareform(pdist(positions))
        vd = np.abs(alts[:, None] - alts[None, :])
        np.fill_diagonal(hd, 50.0)
        np.fill_diagonal(vd, 10000.0)
        return int(np.sum((hd < SEPARATION_MIN) & (vd < get_vertical_min(alts.mean()))) // 2)

    def _compute_hard_los(self, positions, alts):
        """
        Count hard loss-of-separation pairs (horizontal < 5 NM and vertical
        below RVSM hard minimum: 1000 ft below FL290, 2000 ft at/above FL290).
        """
        if len(positions) < 2:
            return 0
        hd = squareform(pdist(positions))
        vd = np.abs(alts[:, None] - alts[None, :])
        np.fill_diagonal(hd, 50.0)
        np.fill_diagonal(vd, 10000.0)
        count = 0
        n = len(alts)
        for i in range(n):
            for j in range(i + 1, n):
                hv = HARD_HIGH if (alts[i] + alts[j]) / 2 >= FL290 else HARD_LOW
                if hd[i, j] < HARD_NM and vd[i, j] < hv:
                    count += 1
        return count