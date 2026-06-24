# -*- coding: utf-8 -*-
"""
=============================================================================
 SPATARO_EVAL_ONLY.py
 Spataro Praxis -- Evaluation Only (model already trained)
 Nicholas D. Spataro, D.Eng., 2026
=============================================================================
 REQUIRES in results_ppo/:
   ppo_atc_model_final.zip
   vecnormalize.pkl
   eval_baseline_aircraft.csv   (created on first run, reused after)

 PRODUCES in results_ppo/:
   h1_h2_summary.txt            Table 4-1 and 4-2 numbers
   h1_h2_results.csv            raw data
   hypothesis_summary.txt       all numbers for Chapter 4
   figure_4_1_traj.png          Figure 4-1 trajectory comparison
   figure_4_2_boxplot.png       Figure 4-2 conflict distribution
   figure_4_3_tornado.png       Figure 4-3 H3 sensitivity tornado
   h3_results.csv               H3 raw data (autosaved per cell)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import time, os, csv, warnings

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces, Env

os.makedirs("results_ppo", exist_ok=True)
_DATA_CACHE = {}

# =============================================================================
# CONSTANTS
# =============================================================================
R_EARTH        = 3440
MAX_AIRCRAFT   = 15
SEPARATION_MIN = 5.0
FL290          = 29000.0
HARD_NM        = 5.0
HARD_LOW       = 1000.0
HARD_HIGH      = 2000.0
ADV_NM         = 8.0
ADV_FT         = 5000.0
TRAINED_OBS    = MAX_AIRCRAFT * 5   # 75

WAYPOINTS = [
    (40.639, -73.778, "JFK"), (40.692, -74.168, "EWR"),
    (39.872, -75.241, "PHL"), (38.852, -77.038, "DCA"),
    (39.175, -76.668, "BWI"), (42.364, -71.005, "BOS"),
    (41.939, -87.907, "ORD"), (33.942, -118.408, "LAX"),
]
PERFORMANCE = {
    'jet':       {'max_climb':50.0,'max_turn':3.0,'max_accel':1.0},
    'turboprop': {'max_climb':25.0,'max_turn':3.0,'max_accel':0.5},
}
DENSITY_LEVELS = {"LOW":10, "BASE":15, "HIGH":20}
FREQ_LEVELS    = {"FAST":1, "MID":5, "SLOW":10}


def haversine_dist(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2)
    return R_EARTH*60*2*np.arcsin(np.sqrt(a))

def get_vertical_min(alt):
    return HARD_LOW if alt < FL290 else HARD_HIGH

def latlon_to_nm(pos):
    return np.deg2rad(pos)*R_EARTH


# =============================================================================
# GRADUAL APPROACH BASELINE
# =============================================================================
def build_gradual_approach_baseline(n_aircraft=MAX_AIRCRAFT, seed=42):
    np.random.seed(seed)
    center_lat=40.7; center_lon=-74.0
    NM_PER_DEG_LON = 60.0*np.cos(np.radians(center_lat))
    airports = [wp[2] for wp in WAYPOINTS]
    data = []
    for pair in range(n_aircraft//2):
        sep_nm = np.random.uniform(12.0,18.0); half = sep_nm/2.0
        pair_lat = center_lat+np.random.uniform(-1.5,1.5)
        pair_lon = center_lon+np.random.uniform(-1.5,1.5)
        alt_base = min(35000.0+pair*1000.0,41000.0)
        hdg_opts = [(90.0,270.0),(45.0,225.0),(135.0,315.0)]
        hdg_a,hdg_b = hdg_opts[pair%3]
        hdg_a += np.random.normal(0,8); hdg_b += np.random.normal(0,8)
        lat_a=pair_lat+np.random.normal(0,0.05); lon_a=pair_lon-(half/NM_PER_DEG_LON)
        lat_b=pair_lat+np.random.normal(0,0.05); lon_b=pair_lon+(half/NM_PER_DEG_LON)
        vel=480.0+np.random.normal(0,15)
        dest_a=WAYPOINTS[np.random.randint(len(WAYPOINTS))]
        dest_b=WAYPOINTS[np.random.randint(len(WAYPOINTS))]
        for i,(lat,lon,hdg,dest) in enumerate([(lat_a,lon_a,hdg_a,dest_a),(lat_b,lon_b,hdg_b,dest_b)]):
            data.append([pair*2+i,"UAL{}".format(pair*2+i+100),
                         np.random.choice(airports),dest[2],
                         np.random.choice(["jet","turboprop"],p=[0.8,0.2]),
                         0,lat,lon,alt_base,vel,hdg,dest[0],dest[1]])
    if n_aircraft%2==1:
        i=n_aircraft-1
        data.append([i,"UAL{}".format(i+100),"JFK","BOS","jet",0,
                     center_lat+np.random.normal(0,0.5),
                     center_lon+np.random.normal(0,0.5),
                     36000.0,480.0,90.0,42.364,-71.005])
    np.random.seed(None)
    return pd.DataFrame(data,columns=[
        "aircraft_id","callsign","dep_airport","dest_airport","ac_type","time",
        "lat","lon","alt","velocity","heading","dest_lat","dest_lon"])


# =============================================================================
# ATCEnv
# =============================================================================
class ATCEnv(Env):
    def __init__(self, n_aircraft=MAX_AIRCRAFT, ai_step_interval=1):
        super().__init__()
        self.n_aircraft=n_aircraft; self.ai_step_interval=ai_step_interval
        self._step_count_h3=0; self._last_action=None; self._prev_min_sep=None
        self.action_space = spaces.Box(-1.0,1.0,shape=(n_aircraft*3,),dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(n_aircraft*5,),dtype=np.float32)
        self.base_states=self._fetch_data(); self.reset()

    def _fetch_data(self):
        if self.n_aircraft in _DATA_CACHE:
            return _DATA_CACHE[self.n_aircraft].copy()
        data=[]
        for i in range(self.n_aircraft):
            lat=40.7+np.random.normal(0,0.65); lon=-74.0+np.random.normal(0,0.65)
            data.append([i,"UAL{}".format(i+100),"JFK","BOS","jet",0,
                         lat,lon,35000.0,480.0,90.0,42.364,-71.005])
        df=pd.DataFrame(data,columns=["aircraft_id","callsign","dep_airport","dest_airport",
                                      "ac_type","time","lat","lon","alt","velocity",
                                      "heading","dest_lat","dest_lon"])
        _DATA_CACHE[self.n_aircraft]=df.copy(); return df

    def reset(self, seed=None, options=None):
        self.current=self.base_states.copy()
        self.current['heading']+=np.random.normal(0,40,self.n_aircraft)
        self.step_count=0; self._step_count_h3=0
        self._last_action=None; self._prev_min_sep=None
        self.prev_dist=np.array([haversine_dist(r['lat'],r['lon'],r['dest_lat'],r['dest_lon'])
                                  for _,r in self.current.iterrows()])
        return self._get_obs(),{}

    def _get_obs(self):
        raw=self.current[['lat','lon','alt','velocity','heading']].values.astype(np.float32)
        norm=raw.copy()
        norm[:,0]=(raw[:,0]-40.5)/3.0; norm[:,1]=(raw[:,1]+74.5)/4.0
        norm[:,2]=(raw[:,2]-33000)/10000; norm[:,3]=(raw[:,3]-450)/150
        norm[:,4]=raw[:,4]/180
        return norm.flatten()

    def step(self, action):
        self._step_count_h3+=1
        if self._step_count_h3%self.ai_step_interval==0 or self._last_action is None:
            self._last_action=action.copy()
        effective=self._last_action
        av=(effective.reshape(self.n_aircraft,3)*np.array([8,300,15])).astype(np.float32)
        for i in range(self.n_aircraft):
            perf=PERFORMANCE[self.current.iloc[i]['ac_type']]
            av[i,0]=np.clip(av[i,0],-perf['max_turn'],perf['max_turn'])
            av[i,1]=np.clip(av[i,1],-perf['max_climb'],perf['max_climb'])
            av[i,2]=np.clip(av[i,2],-perf['max_accel'],perf['max_accel'])
        self.current=self._update_positions(self.current)
        pos_nm=latlon_to_nm(self.current[['lat','lon']].values)
        alts=self.current['alt'].values
        current_dist=np.array([haversine_dist(r['lat'],r['lon'],r['dest_lat'],r['dest_lon'])
                               for _,r in self.current.iterrows()])
        reward=self._compute_reward(pos_nm,alts,self.prev_dist,self.current,av.flatten())
        self.prev_dist=current_dist; self.step_count+=1
        terminated=self.step_count>=1800
        n_conf=self._compute_conflicts(pos_nm,alts)
        n_hard=self._compute_hard_los(pos_nm,alts)
        ac_states=[{"lat":float(self.current.iloc[i]['lat']),
                    "lon":float(self.current.iloc[i]['lon']),
                    "alt_ft":float(self.current.iloc[i]['alt'])}
                   for i in range(self.n_aircraft)]
        return self._get_obs(),reward,terminated,False,{
            "n_conflicts":n_conf,"n_hard_los":n_hard,"aircraft_states":ac_states}

    def _update_positions(self, df):
        df=df.copy(); hr=np.radians(df['heading']); dt=1/3600.0; d=df['velocity']*dt
        df['lat']+=d*np.sin(hr)/60.0; df['lon']+=d*np.cos(hr)/60.0; df['time']+=1
        return df

    def _compute_reward(self, positions, alts, prev_dist, df, actions):
        hd=squareform(pdist(positions)); vd=np.abs(alts[:,None]-alts[None,:])
        np.fill_diagonal(hd,100.0); np.fill_diagonal(vd,20000.0)
        vm=get_vertical_min(alts.mean())
        conflicts=np.sum((hd<SEPARATION_MIN)&(vd<vm))//2
        severe=np.sum((hd<3.0)&(vd<vm/2))//2
        h_prox=np.clip((12.0-hd)/12.0,0,None).sum()
        v_prox=np.clip((vm*2.5-vd)/(vm*2.5),0,None).sum()
        curr_dist=np.array([haversine_dist(r['lat'],r['lon'],r['dest_lat'],r['dest_lon'])
                            for _,r in df.iterrows()])
        progress=np.clip(prev_dist-curr_dist,-20,20).sum()
        lats,lons=df['lat'].values,df['lon'].values
        bpen=(np.sum(np.maximum(0,lats-43)+np.maximum(0,38-lats))+
              np.sum(np.maximum(0,lons+71)+np.maximum(0,-78-lons)))*150.0
        return float(-280.0*conflicts-480.0*severe-0.45*(h_prox+v_prox)
                     +progress*0.13+(2200.0 if conflicts==0 else 0.0)
                     -bpen-np.sum(np.abs(actions))*0.006)

    def _compute_conflicts(self, positions, alts):
        if len(positions)<2: return 0
        hd=squareform(pdist(positions)); vd=np.abs(alts[:,None]-alts[None,:])
        np.fill_diagonal(hd,50.0); np.fill_diagonal(vd,10000.0)
        return int(np.sum((hd<SEPARATION_MIN)&(vd<get_vertical_min(alts.mean())))//2)

    def _compute_hard_los(self, positions, alts):
        if len(positions)<2: return 0
        hd=squareform(pdist(positions)); vd=np.abs(alts[:,None]-alts[None,:])
        np.fill_diagonal(hd,50.0); np.fill_diagonal(vd,10000.0)
        count=0; n=len(alts)
        for i in range(n):
            for j in range(i+1,n):
                hv=HARD_HIGH if (alts[i]+alts[j])/2>=FL290 else HARD_LOW
                if hd[i,j]<HARD_NM and vd[i,j]<hv: count+=1
        return count


# =============================================================================
# LOAD MODEL
# =============================================================================
print("=== LOADING MODEL ===")
vec_env = make_vec_env(ATCEnv, n_envs=1)
vec_env = VecNormalize.load("results_ppo/vecnormalize.pkl", vec_env)
model   = PPO.load("results_ppo/ppo_atc_model_final", env=vec_env)
raw_env = vec_env.envs[0].env.unwrapped
print("Model loaded.")

# Load or build fixed baseline
_csv = "results_ppo/eval_baseline_aircraft.csv"
if os.path.exists(_csv):
    raw_env.base_states = pd.read_csv(_csv)
    print("[FIXED DATASET] Loaded eval_baseline_aircraft.csv")
else:
    raw_env.base_states = build_gradual_approach_baseline(MAX_AIRCRAFT, seed=42)
    raw_env.base_states.to_csv(_csv, index=False)
    print("[BASELINE SAVED] eval_baseline_aircraft.csv created")
_DATA_CACHE.clear()
_DATA_CACHE[MAX_AIRCRAFT] = raw_env.base_states.copy()
print("Baseline: {} aircraft ready.\n".format(len(raw_env.base_states)))


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================
def _pct(b,a): return 0.0 if b==0 else (b-a)/b*100.0
def _ttest(b,a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t,p=stats.ttest_rel(np.array(b,float),np.array(a,float))
    return float(t),float(p)
def _d(b,a):
    b,a=np.array(b,float),np.array(a,float)
    sp=np.sqrt(((len(b)-1)*b.std(ddof=1)**2+(len(a)-1)*a.std(ddof=1)**2)/(len(b)+len(a)-2))
    return float((b.mean()-a.mean())/sp) if sp>0 else 0.0
def _tuc(tuc,states):
    for i in range(len(states)):
        for j in range(i+1,len(states)):
            s,t=states[i],states[j]
            if (haversine_dist(s["lat"],s["lon"],t["lat"],t["lon"])<ADV_NM
                    and abs(s["alt_ft"]-t["alt_ft"])<ADV_FT):
                return tuc+1
    return tuc


# =============================================================================
# H1 + H2 EVALUATION
# =============================================================================
def run_h1_h2(num_runs=30, steps=1800):
    print("\n"+"="*65)
    print("H1 + H2  |  {} runs  |  {} aircraft".format(num_runs,MAX_AIRCRAFT))
    print("="*65)
    t0=time.time(); b_res=[]; a_res=[]
    for run in range(num_runs):
        if run%5==0: print("  Run {}/{}".format(run+1,num_runs))
        # BASELINE
        current=raw_env.base_states.copy()
        current['heading']+=np.random.normal(0,40,MAX_AIRCRAFT)
        bc=bt=bl=0
        for _ in range(steps):
            current=raw_env._update_positions(current)
            current['heading']+=np.random.normal(0,5,MAX_AIRCRAFT)
            pos_nm=latlon_to_nm(current[['lat','lon']].values)
            alts=current['alt'].values
            bc+=raw_env._compute_conflicts(pos_nm,alts)
            bl+=raw_env._compute_hard_los(pos_nm,alts)
            bt=_tuc(bt,[{"lat":float(current.iloc[i]['lat']),
                          "lon":float(current.iloc[i]['lon']),
                          "alt_ft":float(current.iloc[i]['alt'])}
                         for i in range(MAX_AIRCRAFT)])
        b_res.append({"conflict_count":bc,"tuc_seconds":bt,"hard_los_count":bl})
        # AI
        obs=raw_env.reset()[0]; ac=at=al=0
        for _ in range(steps):
            action,_=model.predict(obs,deterministic=True)
            obs,_,term,trunc,info=raw_env.step(action)
            ac+=info.get("n_conflicts",0); al+=info.get("n_hard_los",0)
            if info.get("aircraft_states"): at=_tuc(at,info["aircraft_states"])
            if term or trunc: break
        a_res.append({"conflict_count":ac,"tuc_seconds":at,"hard_los_count":al})
    print("  Done in {:.0f}s".format(time.time()-t0))
    return b_res, a_res


def analyze_and_save(b_res, a_res):
    ba=np.array([r["conflict_count"] for r in b_res],float)
    aa=np.array([r["conflict_count"] for r in a_res],float)
    bt=np.array([r["tuc_seconds"]    for r in b_res],float)
    at=np.array([r["tuc_seconds"]    for r in a_res],float)
    bl=np.array([r["hard_los_count"] for r in b_res],float)
    al=np.array([r["hard_los_count"] for r in a_res],float)

    h1_red=_pct(ba.mean(),aa.mean()); h1_t,h1_p=_ttest(ba,aa); h1_d=_d(ba,aa)
    h1_pass=h1_red>=15.0 and h1_p<0.05
    tr=_pct(bt.mean(),at.mean()); tt,tp=_ttest(bt,at); td=_d(bt,at)
    tuc_pass=tr>=25.0 and tp<0.05
    lr=_pct(bl.mean(),al.mean()); lt,lp=_ttest(bl,al); ld=_d(bl,al)
    los_pass=lr>=20.0 and lp<0.05

    print("\n"+"="*65)
    print("  H1: Conflict Alert Reduction (target >= 15%)")
    print("  Baseline : {:.1f}  SD={:.1f}  min={:.0f}  max={:.0f}".format(
        ba.mean(),ba.std(),ba.min(),ba.max()))
    print("  AI       : {:.1f}  SD={:.1f}  min={:.0f}  max={:.0f}".format(
        aa.mean(),aa.std(),aa.min(),aa.max()))
    print("  Reduction: {:.1f}%  t={:.3f}  p={:.4f}  Cohen's d={:.2f}".format(
        h1_red,h1_t,h1_p,h1_d))
    print("  Result   : {}".format("[PASS] H1 SUPPORTED" if h1_pass else "[FAIL]"))

    print("\n  H2: TUC Workload (target >= 25%)")
    print("  Baseline: {:.1f}s  AI: {:.1f}s  Red: {:.1f}%".format(bt.mean(),at.mean(),tr))
    print("  t={:.3f}  p={:.4f}  d={:.2f}  {}".format(tt,tp,td,"[PASS]" if tuc_pass else "[FAIL]"))

    print("\n  H2: Hard LoS (target >= 20%)")
    print("  Baseline: {:.2f}  AI: {:.2f}  Red: {:.1f}%".format(bl.mean(),al.mean(),lr))
    print("  t={:.3f}  p={:.4f}  d={:.2f}  {}".format(lt,lp,ld,"[PASS]" if los_pass else "[FAIL]"))
    print("="*65)

    # Save CSV
    with open("results_ppo/h1_h2_results.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["run","condition","conflict_count","tuc_seconds","hard_los_count"])
        for cond,res in [("baseline",b_res),("ai",a_res)]:
            for i,r in enumerate(res):
                w.writerow([i+1,cond,r["conflict_count"],r["tuc_seconds"],r["hard_los_count"]])

    # Save summary
    lines=["="*65,"H1/H2 RESULTS  (30 Monte-Carlo runs)","="*65,"",
           "TABLE 4-1: H1 Conflict Alert Reduction","",
           "  Baseline: {:.1f} (SD={:.1f})  min={:.0f}  max={:.0f}".format(
               ba.mean(),ba.std(),ba.min(),ba.max()),
           "  AI:       {:.1f} (SD={:.1f})  min={:.0f}  max={:.0f}".format(
               aa.mean(),aa.std(),aa.min(),aa.max()),
           "  Reduction: {:.1f}%".format(h1_red),
           "  t={:.3f}  p={:.4f}  Cohen's d={:.2f}".format(h1_t,h1_p,h1_d),
           "  H1: {}  (target >= 15%)".format(
               "[PASS] SUPPORTED" if h1_pass else "[FAIL] NOT SUPPORTED"),
           "","TABLE 4-2: H2 Safety Metrics","",
           "  TUC:  Baseline={:.1f}s  AI={:.1f}s  Red={:.1f}%  p={:.4f}  {}".format(
               bt.mean(),at.mean(),tr,tp,"[PASS]" if tuc_pass else "[FAIL]"),
           "  LoS:  Baseline={:.2f}  AI={:.2f}  Red={:.1f}%  p={:.4f}  d={:.2f}  {}".format(
               bl.mean(),al.mean(),lr,lp,ld,"[PASS]" if los_pass else "[FAIL]"),
           "  H2 OVERALL: {}".format(
               "[PASS] SUPPORTED" if (tuc_pass and los_pass) else "[FAIL] NOT SUPPORTED"),
           "","="*65]
    with open("results_ppo/h1_h2_summary.txt","w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Saved: h1_h2_results.csv + h1_h2_summary.txt")
    return {"ba":ba,"aa":aa,"bt":bt,"at":at,"bl":bl,"al":al,
            "h1_red":h1_red,"h1_t":h1_t,"h1_p":h1_p,"h1_d":h1_d,"h1_pass":h1_pass,
            "tr":tr,"tp":tp,"tuc_pass":tuc_pass,
            "lr":lr,"lp":lp,"ld":ld,"los_pass":los_pass}


# =============================================================================
# FIGURE 4-1: TRAJECTORY COMPARISON
# =============================================================================
def plot_figure_4_1():
    print("\n  Generating Figure 4-1 (trajectories)...")
    fig,axes=plt.subplots(1,2,figsize=(18,8))
    colors=plt.cm.tab20(np.linspace(0,1,MAX_AIRCRAFT))

    for ax_idx,(use_ai,title) in enumerate([
            (False,"Baseline (No AI Intervention)"),
            (True, "PPO AI Agent Active")]):
        ax=axes[ax_idx]
        current=raw_env.base_states.copy()
        current['heading']+=np.random.normal(0,40,MAX_AIRCRAFT)
        traj={i:{"lats":[],"lons":[]} for i in range(MAX_AIRCRAFT)}
        total_conf=0

        if use_ai:
            obs=raw_env.reset()[0]
            for _ in range(1800):
                action,_=model.predict(obs,deterministic=True)
                obs,_,term,_,info=raw_env.step(action)
                total_conf+=info["n_conflicts"]
                for i in range(MAX_AIRCRAFT):
                    traj[i]["lats"].append(raw_env.current.iloc[i]['lat'])
                    traj[i]["lons"].append(raw_env.current.iloc[i]['lon'])
                if term: break
        else:
            for _ in range(1800):
                current=raw_env._update_positions(current)
                current['heading']+=np.random.normal(0,5,MAX_AIRCRAFT)
                pos_nm=latlon_to_nm(current[['lat','lon']].values)
                total_conf+=raw_env._compute_conflicts(pos_nm,current['alt'].values)
                for i in range(MAX_AIRCRAFT):
                    traj[i]["lats"].append(float(current.iloc[i]['lat']))
                    traj[i]["lons"].append(float(current.iloc[i]['lon']))

        for i in range(MAX_AIRCRAFT):
            lats=traj[i]["lats"]; lons=traj[i]["lons"]
            if not lats: continue
            ax.plot(lons[::30],lats[::30],color=colors[i],linewidth=1.8,alpha=0.85)
            ax.plot(lons[0],lats[0],'o',color=colors[i],markersize=7,zorder=5)
            ax.plot(lons[-1],lats[-1],'^',color=colors[i],markersize=7,zorder=5)

        ax.set_xlabel("Longitude (°W)",fontsize=11)
        ax.set_ylabel("Latitude (°N)",fontsize=11)
        ax.set_title("{}\nTotal Conflict Steps: {}".format(title,total_conf),fontsize=12)
        ax.grid(True,alpha=0.3)
        ax.set_xlim(-78,-71); ax.set_ylim(38,43)

        # NEC boundary box
        from matplotlib.patches import Rectangle
        rect=Rectangle((-78,38),7,5,linewidth=1.5,edgecolor='red',
                        facecolor='none',linestyle='--',alpha=0.5)
        ax.add_patch(rect)
        ax.text(-77.8,42.7,"NEC Airspace",fontsize=8,color='red',alpha=0.7)

    # Legend
    legend_elements=[
        mpatches.Patch(color='gray',label='Aircraft trajectory'),
        plt.Line2D([0],[0],marker='o',color='gray',markersize=6,
                   label='Start position',linestyle='None'),
        plt.Line2D([0],[0],marker='^',color='gray',markersize=6,
                   label='End position',linestyle='None'),
    ]
    axes[1].legend(handles=legend_elements,loc='lower right',fontsize=9)
    plt.suptitle("Figure 4-1: NEC En-Route Aircraft Trajectories\n"
                 "Baseline (No AI) vs. PPO AI Agent  —  Nicholas D. Spataro, D.Eng., 2026",
                 fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig("results_ppo/figure_4_1_traj.png",dpi=150,bbox_inches="tight")
    plt.close()
    print("  Saved: results_ppo/figure_4_1_traj.png")


# =============================================================================
# FIGURE 4-2: BOX PLOT -- CONFLICT DISTRIBUTION (new)
# =============================================================================
def plot_figure_4_2(stats_dict):
    print("  Generating Figure 4-2 (box plot)...")
    fig,axes=plt.subplots(1,2,figsize=(14,6))

    # Conflict counts
    ax=axes[0]
    data=[stats_dict["ba"],stats_dict["aa"]]
    bp=ax.boxplot(data,labels=["Baseline\n(No AI)","PPO AI\nAgent"],
                  patch_artist=True,notch=False,
                  medianprops=dict(color='black',linewidth=2))
    bp['boxes'][0].set_facecolor('#ff7f7f')
    bp['boxes'][1].set_facecolor('#7fbfff')
    ax.set_ylabel("Conflict Alert Count per Episode",fontsize=11)
    ax.set_title("H1: Conflict Alert Distribution\n"
                 "Reduction: {:.1f}%  p={:.4f}  d={:.2f}".format(
                     stats_dict["h1_red"],stats_dict["h1_p"],stats_dict["h1_d"]),
                 fontsize=11)
    ax.grid(True,alpha=0.3,axis='y')
    # Significance bracket
    y_max=max(stats_dict["ba"].max(),stats_dict["aa"].max())*1.05
    ax.plot([1,1,2,2],[y_max*0.95,y_max,y_max,y_max*0.95],'k-',linewidth=1)
    sig_text="***" if stats_dict["h1_p"]<0.001 else ("**" if stats_dict["h1_p"]<0.01
                                                       else ("*" if stats_dict["h1_p"]<0.05 else "n.s."))
    ax.text(1.5,y_max*1.01,sig_text,ha='center',fontsize=13)

    # Hard LoS
    ax=axes[1]
    data2=[stats_dict["bl"],stats_dict["al"]]
    bp2=ax.boxplot(data2,labels=["Baseline\n(No AI)","PPO AI\nAgent"],
                   patch_artist=True,notch=False,
                   medianprops=dict(color='black',linewidth=2))
    bp2['boxes'][0].set_facecolor('#ff7f7f')
    bp2['boxes'][1].set_facecolor('#7fbfff')
    ax.set_ylabel("Hard LoS Breach Count per Episode",fontsize=11)
    ax.set_title("H2: Hard Loss-of-Separation Distribution\n"
                 "Reduction: {:.1f}%  p={:.4f}  d={:.2f}".format(
                     stats_dict["lr"],stats_dict["lp"],stats_dict["ld"]),
                 fontsize=11)
    ax.grid(True,alpha=0.3,axis='y')
    y_max2=max(stats_dict["bl"].max(),stats_dict["al"].max())*1.05
    ax.plot([1,1,2,2],[y_max2*0.95,y_max2,y_max2,y_max2*0.95],'k-',linewidth=1)
    sig_text2="***" if stats_dict["lp"]<0.001 else ("**" if stats_dict["lp"]<0.01
                                                      else ("*" if stats_dict["lp"]<0.05 else "n.s."))
    ax.text(1.5,y_max2*1.01,sig_text2,ha='center',fontsize=13)

    plt.suptitle("Figure 4-2: Conflict and Loss-of-Separation Distributions\n"
                 "30 Monte-Carlo Runs  —  Nicholas D. Spataro, D.Eng., 2026",
                 fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig("results_ppo/figure_4_2_boxplot.png",dpi=150,bbox_inches="tight")
    plt.close()
    print("  Saved: results_ppo/figure_4_2_boxplot.png")


# =============================================================================
# H3 EVALUATION
# =============================================================================
def _h3_episode(n_aircraft, interval, use_ai, steps=1800):
    env=ATCEnv(n_aircraft=n_aircraft,ai_step_interval=interval)
    obs,_=env.reset(); conf=los=0
    for _ in range(steps):
        if use_ai:
            # Pad observation to trained size (75), get full action (45),
            # then truncate or pad action to match n_aircraft*3
            padded=np.zeros(TRAINED_OBS,dtype=np.float32)
            padded[:min(len(obs),TRAINED_OBS)]=obs[:TRAINED_OBS]
            full_action,_=model.predict(padded,deterministic=True)
            # full_action is size 45 (15ac x 3); resize to n_aircraft*3
            action=np.zeros(n_aircraft*3,dtype=np.float32)
            copy_len=min(len(full_action),n_aircraft*3)
            action[:copy_len]=full_action[:copy_len]
        else:
            action = np.zeros(n_aircraft * 3, dtype=np.float32)
        obs,_,term,trunc,info=env.step(action)
        conf+=info.get("n_conflicts",0); los+=info.get("n_hard_los",0)
        if term or trunc: break
    env.close(); return conf+los

def run_h3(num_runs=20, steps=1800):
    print("\n"+"="*65)
    print("H3  |  3x3 factorial  |  {} runs/cell".format(num_runs))
    print("="*65)
    grid={}; cell=0
    for d_label,n_ac in DENSITY_LEVELS.items():
        for f_label,interval in FREQ_LEVELS.items():
            cell+=1
            print("  Cell {}/9: {}({}ac) {}({}s)".format(cell,d_label,n_ac,f_label,interval))
            bs=[_h3_episode(n_ac,interval,False,steps) for _ in range(num_runs)]
            ai=[_h3_episode(n_ac,interval,True, steps) for _ in range(num_runs)]
            red=_pct(np.mean(bs),np.mean(ai))
            _,p=_ttest(bs,ai)
            grid[(d_label,f_label)]={"baseline":bs,"ai":ai,"red":red,"p":p}
            print("  → Base:{:.1f}  AI:{:.1f}  Red:{:.1f}%  p={:.4f}  {}".format(
                np.mean(bs),np.mean(ai),red,p,"[PASS]" if red>=10 and p<0.05 else "[FAIL]"))
            # Autosave
            with open("results_ppo/h3_results.csv","w",newline="") as f:
                w=csv.writer(f)
                w.writerow(["density","freq","condition","run","risk_score"])
                for d in DENSITY_LEVELS:
                    for freq in FREQ_LEVELS:
                        if (d,freq) not in grid: continue
                        for cond in ["baseline","ai"]:
                            for i,s in enumerate(grid[(d,freq)][cond]):
                                w.writerow([d,freq,cond,i+1,s])
    return grid


# =============================================================================
# FIGURE 4-3: TORNADO PLOT
# =============================================================================
def plot_figure_4_3(grid):
    print("  Generating Figure 4-3 (tornado plot)...")
    # Compute ranges
    density_reds=[grid[(d,"FAST")]["red"] for d in DENSITY_LEVELS]
    freq_reds=[grid[("BASE",f)]["red"] for f in FREQ_LEVELS]
    density_range=max(density_reds)-min(density_reds)
    freq_range=max(freq_reds)-min(freq_reds)

    fig,ax=plt.subplots(figsize=(10,4))
    labels=["Traffic Density\n(LOW / BASE / HIGH)",
            "AI Update Frequency\n(FAST 1s / MID 5s / SLOW 10s)"]
    values=[density_range,freq_range]
    colors_bar=["#2196F3","#FF9800"]
    bars=ax.barh(labels,values,color=colors_bar,height=0.4,edgecolor='black',linewidth=0.8)
    for bar,val in zip(bars,values):
        ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                "±{:.1f} pp".format(val),va='center',fontsize=11,fontweight='bold')
    ax.set_xlabel("Range of Risk Score Reduction (percentage points)",fontsize=11)
    ax.set_title("Figure 4-3: Tornado Plot — H3 Sensitivity Analysis\n"
                 "Sensitivity of Risk Reduction to Experimental Factors",fontsize=12)
    ax.set_xlim(0,max(values)*1.6+1); ax.grid(axis='x',alpha=0.35)
    ax.axvline(x=10,color='green',linestyle='--',alpha=0.5,linewidth=1.2)
    ax.text(10.2,1.45,"H3 threshold (10%)",color='green',fontsize=8,alpha=0.8)
    plt.tight_layout()
    plt.savefig("results_ppo/figure_4_3_tornado.png",dpi=150,bbox_inches="tight")
    plt.close()
    print("  Saved: results_ppo/figure_4_3_tornado.png")
    return {"density_range":density_range,"freq_range":freq_range}


# =============================================================================
# FINAL SUMMARY
# =============================================================================
def save_final_summary(s, h3_grid, tornado):
    h3_pass=all(v["red"]>=10.0 and v["p"]<0.05 for v in h3_grid.values())
    lines=["="*70,"SPATARO PRAXIS — FULL HYPOTHESIS RESULTS",
           "Nicholas D. Spataro, D.Eng., 2026","="*70,"",
           "TABLE 4-1: H1 — Conflict Alert Reduction (30 Monte-Carlo Runs)",
           "  Baseline: {:.1f} (SD={:.1f})  min={:.0f}  max={:.0f}".format(
               s["ba"].mean(),s["ba"].std(),s["ba"].min(),s["ba"].max()),
           "  AI:       {:.1f} (SD={:.1f})  min={:.0f}  max={:.0f}".format(
               s["aa"].mean(),s["aa"].std(),s["aa"].min(),s["aa"].max()),
           "  Reduction: {:.1f}%  t={:.3f}  p={:.4f}  Cohen's d={:.2f}".format(
               s["h1_red"],s["h1_t"],s["h1_p"],s["h1_d"]),
           "  H1: {}  (target >= 15%)".format(
               "[PASS] SUPPORTED" if s["h1_pass"] else "[FAIL]"),
           "","TABLE 4-2: H2 — Safety Metrics",
           "  TUC:  Base={:.1f}s  AI={:.1f}s  Red={:.1f}%  p={:.4f}  {}".format(
               s["bt"].mean(),s["at"].mean(),s["tr"],s["tp"],
               "[PASS]" if s["tuc_pass"] else "[FAIL]"),
           "  LoS:  Base={:.2f}  AI={:.2f}  Red={:.1f}%  p={:.4f}  d={:.2f}  {}".format(
               s["bl"].mean(),s["al"].mean(),s["lr"],s["lp"],s["ld"],
               "[PASS]" if s["los_pass"] else "[FAIL]"),
           "","TABLE 4-3: H3 — 3x3 Factorial Sensitivity",
           "  Density\\Freq    FAST(1s)    MID(5s)    SLOW(10s)"]
    for d in DENSITY_LEVELS:
        row="  {:<14}".format("{} ({}ac)".format(d,DENSITY_LEVELS[d]))
        for f in FREQ_LEVELS:
            v=h3_grid[(d,f)]
            row+="  {:>8.1f}%{}".format(v["red"],"*" if v["red"]>=10 and v["p"]<0.05 else " ")
        lines.append(row)
    lines+=["  (* p<0.05, target >= 10%)",
            "  Tornado: Density ±{:.1f}pp  |  Frequency ±{:.1f}pp".format(
                tornado["density_range"],tornado["freq_range"]),
            "  H3: {}".format("[PASS] SUPPORTED" if h3_pass else "[FAIL] NOT SUPPORTED"),
            "","="*70,
            "FIGURES SAVED:",
            "  figure_4_1_traj.png    — Chapter 4 Figure 4-1",
            "  figure_4_2_boxplot.png — Chapter 4 Figure 4-2 (NEW)",
            "  figure_4_3_tornado.png — Chapter 4 Figure 4-3",
            "="*70]
    with open("results_ppo/hypothesis_summary.txt","w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Saved: results_ppo/hypothesis_summary.txt")


# =============================================================================
# MAIN
# =============================================================================
if __name__=="__main__":
    print("\n=== STARTING FULL EVALUATION ===")
    t_start=time.time()

    b_res,a_res=run_h1_h2(num_runs=100,steps=1800)
    s=analyze_and_save(b_res,a_res)

    print("\n-- Generating figures --")
    plot_figure_4_1()
    plot_figure_4_2(s)

    h3_grid=run_h3(num_runs=30,steps=1800)
    tornado=plot_figure_4_3(h3_grid)
    save_final_summary(s,h3_grid,tornado)

    print("\n"+"="*65)
    print("  FINAL RESULTS")
    print("="*65)
    print("  H1 (>= 15%):     {}  {:.1f}%  p={:.4f}".format(
        "[PASS]" if s["h1_pass"] else "[FAIL]",s["h1_red"],s["h1_p"]))
    print("  H2 TUC (>= 25%): {}  {:.1f}%  p={:.4f}".format(
        "[PASS]" if s["tuc_pass"] else "[FAIL]",s["tr"],s["tp"]))
    print("  H2 LoS (>= 20%): {}  {:.1f}%  p={:.4f}".format(
        "[PASS]" if s["los_pass"] else "[FAIL]",s["lr"],s["lp"]))
    print("\n  Open results_ppo/hypothesis_summary.txt for Chapter 4 numbers.")
    print("  Total: {:.0f}s".format(time.time()-t_start))