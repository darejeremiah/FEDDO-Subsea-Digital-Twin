"""Educational subsea pipeline digital twin. Not for safety-critical decisions."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

DB = Path(__file__).with_name('pipeline_historian.sqlite')

@dataclass
class Pipeline:
    outside_diameter_m: float = 0.3239
    nominal_wall_m: float = 0.0206
    minimum_wall_m: float = 0.0120
    span_m: float = 18.0
    elastic_modulus_pa: float = 207e9
    steel_density_kg_m3: float = 7850.0
    water_density_kg_m3: float = 1025.0
    fluid_density_kg_m3: float = 850.0
    added_mass_coefficient: float = 1.0
    strouhal: float = 0.2
    thermal_expansion_per_k: float = 12e-6
    reference_temperature_c: float = 20.0
    allowable_hoop_stress_pa: float = 250e6
    sn_a: float = 1e12  # illustrative cycles at 1 MPa, slope m=3
    sn_m: float = 3.0
    corrosion_rate_mm_y: float = 0.20


def evaluate(p: Pipeline, pressure_bar: float, temp_c: float, current_m_s: float,
             thickness_mm: float, vibration_hz: float, vibration_stress_mpa: float,
             elapsed_hours: float = 1.0, prior_damage: float = 0.0) -> dict:
    """Screening calculations; frequency is simply supported beam estimate."""
    d = p.outside_diameter_m
    t = thickness_mm / 1000
    if not 0 < t < d / 2:
        raise ValueError('Wall thickness must be between zero and half the outside diameter.')
    di = d - 2*t
    area = np.pi/4*(d*d-di*di)
    inertia = np.pi/64*(d**4-di**4)
    steel_mass = p.steel_density_kg_m3*area
    contents_mass = p.fluid_density_kg_m3*np.pi/4*di**2
    added_mass = p.added_mass_coefficient*p.water_density_kg_m3*np.pi/4*d**2
    mass_per_m = steel_mass+contents_mass+added_mass
    natural_hz = np.pi/(2*p.span_m**2)*np.sqrt(p.elastic_modulus_pa*inertia/mass_per_m)
    shedding_hz = p.strouhal*max(current_m_s,0)/d
    ratio = shedding_hz/natural_hz
    hoop_mpa = pressure_bar*1e5*d/(2*t)/1e6  # thin-wall screening
    thermal_mpa = p.elastic_modulus_pa*p.thermal_expansion_per_k*(temp_c-p.reference_temperature_c)/1e6
    # Thermal stress is an upper-bound axial estimate for full restraint.
    cycles = max(vibration_hz,0)*elapsed_hours*3600
    # Illustrative S-N relation, stress range in MPa; NOT a code-qualified curve.
    allowable_cycles = p.sn_a/max(vibration_stress_mpa,0.001)**p.sn_m
    damage_increment = cycles/allowable_cycles
    damage = prior_damage+damage_increment
    remaining_wall_mm = thickness_mm-p.minimum_wall_m*1000
    corrosion_life_y = max(0,remaining_wall_mm)/p.corrosion_rate_mm_y if p.corrosion_rate_mm_y>0 else float('inf')
    pressure_util = hoop_mpa/(p.allowable_hoop_stress_pa/1e6)
    warnings=[]
    if thickness_mm <= p.minimum_wall_m*1000: warnings.append('Wall thickness at/below configured minimum')
    if pressure_util >= 1: warnings.append('Hoop stress above configured screening limit')
    if 0.8 <= ratio <= 1.2: warnings.append('Vortex shedding near first bending frequency: detailed VIV assessment required')
    if damage >= 1: warnings.append('Illustrative fatigue damage >= 1')
    if corrosion_life_y < 5: warnings.append('Corrosion remaining life below 5 years')
    return dict(hoop_stress_mpa=hoop_mpa, thermal_stress_mpa=thermal_mpa,
                first_natural_frequency_hz=natural_hz, shedding_frequency_hz=shedding_hz,
                frequency_ratio=ratio, pressure_utilization=pressure_util,
                fatigue_damage_increment=damage_increment, fatigue_damage=damage,
                corrosion_remaining_life_years=corrosion_life_y,
                corrosion_rate_mm_y=p.corrosion_rate_mm_y, warnings=warnings)


def simulate(p: Pipeline, hours: int, seed: int) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    ts=pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h'),periods=hours,freq='h')
    x=np.arange(hours)
    pressure=125+8*np.sin(x/11)+rng.normal(0,2,hours)
    temperature=54+4*np.sin(x/17)+rng.normal(0,.8,hours)
    current=np.maximum(.05, .65+.24*np.sin(x/13)+rng.normal(0,.07,hours))
    thickness=p.nominal_wall_m*1000-p.corrosion_rate_mm_y*x/(365.25*24)
    thickness+=rng.normal(0,.015,hours)
    vibration_stress=np.maximum(1,16+8*current+rng.normal(0,2,hours))
    vibration_hz=np.maximum(.01,p.strouhal*current/p.outside_diameter_m+rng.normal(0,.02,hours))
    df=pd.DataFrame(dict(timestamp=ts,pressure_bar=pressure,temperature_c=temperature,
        current_m_s=current,thickness_mm=thickness,vibration_hz=vibration_hz,
        vibration_stress_mpa=vibration_stress))
    # A demonstrative abnormal pressure event near the end.
    if hours>30: df.loc[hours-12:hours-9,'pressure_bar']+=35
    return df


def assess_series(df: pd.DataFrame,p: Pipeline) -> pd.DataFrame:
    rows=[]
    damage=0.0
    for r in df.itertuples(index=False):
        result=evaluate(p,r.pressure_bar,r.temperature_c,r.current_m_s,
                        r.thickness_mm,r.vibration_hz,r.vibration_stress_mpa,
                        elapsed_hours=1,prior_damage=damage)
        damage=result['fatigue_damage']
        rows.append(result)
    out=pd.concat([df.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    features=['pressure_bar','temperature_c','current_m_s','thickness_mm','vibration_hz','vibration_stress_mpa']
    if len(out)>=30:
        model=IsolationForest(n_estimators=120,contamination=.04,random_state=42)
        out['anomaly']=model.fit_predict(out[features])==-1
    else: out['anomaly']=False
    out['alarm_count']=out['warnings'].map(len)
    return out


def save_history(df:pd.DataFrame) -> None:
    db=df.copy()
    db['timestamp']=db['timestamp'].astype(str)
    db['warnings']=db['warnings'].apply(json.dumps)
    with sqlite3.connect(DB) as conn:
        db.to_sql('sensor_and_twin_history',conn,if_exists='replace',index=False)

st.set_page_config(page_title='Subsea Pipeline Digital Twin',layout='wide',page_icon='🌊')
st.title('🌊 Subsea Pipeline Integrity Digital Twin')
st.caption('Synthetic-data engineering demonstration | Not certified for operational or safety-critical use')
with st.sidebar:
    st.header('Pipeline configuration')
    p=Pipeline()
    p.outside_diameter_m=st.number_input('Outside diameter (m)',.05,2.0,p.outside_diameter_m)
    p.nominal_wall_m=st.number_input('Nominal wall thickness (mm)',5.,80.,p.nominal_wall_m*1000)/1000
    p.minimum_wall_m=st.number_input('Minimum wall thickness (mm)',1.,60.,p.minimum_wall_m*1000)/1000
    p.span_m=st.number_input('Free span (m)',1.,150.,p.span_m)
    p.corrosion_rate_mm_y=st.number_input('Corrosion rate (mm/year)',0.,5.,p.corrosion_rate_mm_y)
    p.allowable_hoop_stress_pa=st.number_input('Hoop screening limit (MPa)',50.,700.,p.allowable_hoop_stress_pa/1e6)*1e6
    hours=st.slider('Simulation history (hours)',48,1000,240,24)
    seed=st.number_input('Simulation seed',0,100000,42)
    uploaded=st.file_uploader('Optional sensor CSV',type='csv')
    st.caption('CSV columns: timestamp, pressure_bar, temperature_c, current_m_s, thickness_mm, vibration_hz, vibration_stress_mpa')

if p.minimum_wall_m>=p.nominal_wall_m:
    st.error('Minimum wall thickness must be less than nominal wall thickness.')
    st.stop()
if uploaded is None:
    raw=simulate(p,hours,int(seed))
else:
    raw=pd.read_csv(uploaded)
    required=['timestamp','pressure_bar','temperature_c','current_m_s','thickness_mm','vibration_hz','vibration_stress_mpa']
    missing=set(required)-set(raw.columns)
    if missing:
        st.error(f'Missing CSV columns: {sorted(missing)}')
        st.stop()
    raw=raw[required].copy()
    raw['timestamp']=pd.to_datetime(raw['timestamp'],utc=True,errors='coerce')
    for c in required[1:]: raw[c]=pd.to_numeric(raw[c],errors='coerce')
    raw=raw.dropna().sort_values('timestamp').drop_duplicates('timestamp')
    if raw.empty or (raw['thickness_mm']<=0).any() or (raw['thickness_mm']>=p.outside_diameter_m*500).any():
        st.error('Invalid or empty sensor data, including invalid thickness values.')
        st.stop()

out=assess_series(raw,p)
latest=out.iloc[-1]
c1,c2,c3,c4,c5=st.columns(5)
c1.metric('Pressure',f"{latest['pressure_bar']:.1f} bar")
c2.metric('Temperature',f"{latest['temperature_c']:.1f} °C")
c3.metric('Wall thickness',f"{latest['thickness_mm']:.2f} mm")
c4.metric('Corrosion life',f"{latest['corrosion_remaining_life_years']:.1f} y")
c5.metric('Fatigue damage',f"{latest['fatigue_damage']:.5f}")
if latest['warnings']:
    st.warning('Latest engineering warnings: '+'; '.join(latest['warnings']))
else: st.success('No configured screening limits exceeded in latest sample.')
if latest['anomaly']: st.warning('ML anomaly flag on latest sample: investigate sensor and process data.')

tabs=st.tabs(['Pressure & temperature','Corrosion','VIV & fatigue','Alerts & AI','Data & export'])
def chart(cols,title):
    fig=go.Figure()
    for col in cols: fig.add_trace(go.Scatter(x=out['timestamp'],y=out[col],name=col,mode='lines'))
    fig.update_layout(title=title,xaxis_title='UTC timestamp',height=420,legend=dict(orientation='h'))
    return fig
with tabs[0]:
    st.plotly_chart(chart(['pressure_bar','temperature_c'],'Pressure (bar) and temperature (°C) — separate units'),use_container_width=True)
    st.plotly_chart(chart(['hoop_stress_mpa','thermal_stress_mpa'],'Hoop and fully restrained thermal stress (MPa)'),use_container_width=True)
with tabs[1]:
    st.plotly_chart(chart(['thickness_mm'],'Measured / simulated wall thickness (mm)'),use_container_width=True)
    st.plotly_chart(chart(['corrosion_remaining_life_years'],'Corrosion screening remaining life (years)'),use_container_width=True)
    st.info('Remaining life = (current thickness − configured minimum thickness) / corrosion rate. Assumes constant uniform corrosion; local pitting is not modelled.')
with tabs[2]:
    st.plotly_chart(chart(['first_natural_frequency_hz','shedding_frequency_hz','vibration_hz'],'VIV frequencies (Hz)'),use_container_width=True)
    st.plotly_chart(chart(['fatigue_damage'],'Cumulative illustrative Miner damage'),use_container_width=True)
    st.info('VIV resonance is screened by shedding/first-natural frequency ratio 0.8–1.2. Fatigue uses an illustrative S–N curve and measured/synthetic stress range.')
with tabs[3]:
    st.metric('ML flagged samples',int(out['anomaly'].sum()))
    st.metric('Samples with engineering warnings',int((out['alarm_count']>0).sum()))
    alerts=out.loc[(out['alarm_count']>0)|out['anomaly'],['timestamp','warnings','anomaly','pressure_utilization','frequency_ratio']].copy()
    alerts['warnings']=alerts['warnings'].apply('; '.join)
    st.dataframe(alerts.tail(100),use_container_width=True)
    st.caption('Isolation Forest is trained on the selected history. Its flags are statistical outliers, not validated failure probabilities.')
with tabs[4]:
    st.dataframe(out.tail(100),use_container_width=True)
    csv=out.copy()
    csv['warnings']=csv['warnings'].apply(json.dumps)
    st.download_button('Download evaluated CSV',csv.to_csv(index=False).encode('utf-8'),'subsea_twin_results.csv','text/csv')
    st.download_button('Download pipeline configuration',json.dumps(asdict(p),indent=2),'pipeline_config.json','application/json')
    if st.button('Save evaluated history to local SQLite'):
        save_history(out)
        st.success(f'History saved to {DB.name} (replaces previous table).')

st.divider()
st.caption('Engineering limitations: simplified thin-wall hoop stress, fully restrained thermal stress, simply supported Euler–Bernoulli span frequency, Strouhal VIV screening, and illustrative fatigue S–N. No multiphase flow, detailed corrosion chemistry, soil interaction, buckling, weld SCFs, rainflow counting, sensor uncertainty or code-qualified acceptance checks.')
