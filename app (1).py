import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="FEDDO GROUP | Subsea Digital Twin V2.0",layout="wide")
st.title("🌊 FEDDO GROUP — Subsea Pipeline Digital Twin V2.0")
st.caption("Engineering demonstration • Synthetic spatial model • Not certified for operational or safety-critical decisions")
with st.sidebar:
    st.header("Pipeline configuration")
    D=st.number_input("Outside diameter (m)",0.1,1.5,0.324,format="%.3f")
    t0=st.number_input("Nominal thickness (mm)",3.,60.,20.6)
    tmin=st.number_input("Minimum screening thickness (mm)",1.,60.,14.)
    depth=st.number_input("Water depth (m)",0.,4000.,1000.)
    L=st.slider("Free span (m)",5,80,30)
    age=st.slider("Asset age (years)",0,30,5)
    cr=st.slider("Base corrosion rate (mm/year)",0.,2.,0.4)
    P=st.slider("Inlet gauge pressure (bar)",0,350,135)
    T=st.slider("Inlet temperature (°C)",4,150,65)
    U=st.slider("Current velocity (m/s)",0.,3.,0.8)
    amp=st.slider("Assumed peak VIV amplitude (mm)",0.,50.,5.)
    restraint=st.slider("Thermal restraint fraction",0.,1.,0.5)
    hours=st.slider("History (hours)",48,720,240)
    event=st.checkbox("Inject pressure excursion",True)
    seed=st.number_input("Random seed",0,99999,42)
if tmin>=t0 or 2*t0/1000>=D:
    st.error("Require minimum thickness below nominal thickness and positive bore."); st.stop()
E=207e9; alpha=12e-6
pe=1025*9.81*depth/1e5
x=np.linspace(0,500,201)
hot=np.exp(-((x-180)/30)**2)
span=np.sin(np.pi*np.clip((x-(300-L/2))/L,0,1))
rate=cr*(1+1.5*hot)
thick=t0-rate*age
if np.any(thick<=0):
    st.error("Model predicts through-wall loss. Calculations stopped."); st.stop()
rng=np.random.default_rng(int(seed))
h=np.arange(hours)
pressure=P+5*np.sin(h/12)+rng.normal(0,1,hours)
temp=T+2*np.sin(h/19)+rng.normal(0,.4,hours)
current=np.maximum(0,U+.08*np.sin(h/9)+rng.normal(0,.025,hours))
if event: pressure[-12:-8]+=35
upload=st.sidebar.file_uploader("Optional sensor history CSV",type="csv")
if upload:
    try:
        df=pd.read_csv(upload)
        cols=["timestamp","pressure_bar","temperature_c","current_m_s"]
        df=df[cols].copy()
        df["timestamp"]=pd.to_datetime(df.timestamp,utc=True,errors="raise")
        for c in cols[1:]: df[c]=pd.to_numeric(df[c],errors="raise")
        df=df.sort_values("timestamp").reset_index(drop=True)
        if len(df)<48 or df.timestamp.duplicated().any() or not np.isfinite(df[cols[1:]].values).all() or (df.current_m_s<0).any(): raise ValueError("Need 48+ unique finite samples and nonnegative currents")
    except Exception as e:
        st.error(f"Invalid CSV: {e}"); st.stop()
else:
    df=pd.DataFrame({"timestamp":pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"),periods=hours,freq="h"),"pressure_bar":pressure,"temperature_c":temp,"current_m_s":current})
dt=df.timestamp.diff().dt.total_seconds().fillna(0).to_numpy()
# First sample contributes no elapsed exposure; large gaps excluded.
typical=np.median(dt[1:]); dt=np.where(dt<=3*typical,dt,0)
def natural(t):
    di=D-2*t/1000
    I=np.pi*(D**4-di**4)/64
    mass=7850*np.pi*(D**2-di**2)/4+850*np.pi*di**2/4+1025*np.pi*D**2/4
    return np.pi/(2*L**2)*np.sqrt(E*I/mass)
fn=natural(thick[np.argmin(abs(x-300))])
df["hoop_mpa"]=(df.pressure_bar-pe)*1e5*D/(2*thick.min()/1000)/1e6
df["thermal_mpa"]=-restraint*E*alpha*(df.temperature_c-4)/1e6
df["shedding_hz"]=.2*df.current_m_s/D
df["frequency_ratio"]=df.shedding_hz/fn
# Prescribed response envelope, not a solved hydrodynamic response.
response=np.exp(-((df.frequency_ratio-1)/.25)**2)
df["amplitude_mm"]=amp*response
df["stress_range_mpa"]=2*E*(D/2)*(np.pi/L)**2*df.amplitude_mm/1000/1e6
df["fatigue_damage"]=np.cumsum(fn*dt*df.stress_range_mpa**3/1e12)
features=["pressure_bar","temperature_c","current_m_s"]
n=max(24,len(df)//3)
model=IsolationForest(n_estimators=100,contamination=.03,random_state=int(seed)).fit(df[features].iloc[:n])
df["ai_anomaly"]=model.predict(df[features])==-1
df["ai_score"]=model.decision_function(df[features])
latest=df.iloc[-1]
rl=np.divide(np.maximum(thick-tmin,0),rate,out=np.full_like(rate,np.inf),where=rate>0)
rl=np.where(thick<=tmin,0,rl)
sp=pd.DataFrame({"distance_m":x,"metal_loss_mm":t0-thick,"thickness_mm":thick,"remaining_life_years":rl,"pressure_bar":latest.pressure_bar-15*x/500,"temperature_c":4+(latest.temperature_c-4)*np.exp(-x/800),"viv_mm":latest.amplitude_mm*span,"fatigue_damage":latest.fatigue_damage*span**3})
for c,v in zip(st.columns(5),[("Pressure",f"{latest.pressure_bar:.1f} bar"),("Temperature",f"{latest.temperature_c:.1f} °C"),("Minimum wall",f"{thick.min():.2f} mm"),("Corrosion life",f"{rl.min():.1f} y"),("Span fatigue",f"{latest.fatigue_damage:.3g}")]): c.metric(*v)
alerts=[]
if thick.min()<=tmin: alerts.append("Minimum thickness reached")
if latest.pressure_bar>160: alerts.append("Pressure exceeds illustrative limit")
if abs(latest.hoop_mpa)>250: alerts.append("Hoop stress screening limit exceeded")
if latest.temperature_c>80: alerts.append("Temperature exceeds illustrative limit")
if .8<=latest.frequency_ratio<=1.2: alerts.append("Potential VIV resonance")
if latest.fatigue_damage>=1: alerts.append("Nominal Miner damage limit reached")
if latest.ai_anomaly: alerts.append("AI anomaly")
if latest.pressure_bar<pe: alerts.append("Net external pressure: collapse not assessed")
if alerts: st.warning(" • ".join(alerts))
else: st.info("No configured latest-sample screening flags; not a safety determination.")
tabs=st.tabs(["3D virtual asset","Pressure & temperature","Corrosion","VIV & fatigue","AI & alerts","Data & export"])
with tabs[0]:
    fields={"Corrosion metal loss (mm)":"metal_loss_mm","Wall thickness (mm)":"thickness_mm","Pressure (bar)":"pressure_bar","Temperature (°C)":"temperature_c","VIV amplitude (mm)":"viv_mm","Fatigue damage":"fatigue_damage"}
    choice=st.selectbox("Colour pipeline by",list(fields))
    animate=st.checkbox("Enable VIV animation")
    kp=st.slider("Virtual inspection station (m)",0,500,180,5)
    i=np.argmin(abs(x-kp)); st.dataframe(sp.iloc[[i]],hide_index=True)
    theta=np.linspace(0,2*np.pi,24)
    X,TH=np.meshgrid(x,theta)
    # Exaggerated cylindrical radius and span clearance for visibility.
    radius=1.2; bed=-depth+0.8*np.sin(x/45)
    zcenter=bed+radius+2.5*span
    Y=radius*np.cos(TH); Z=zcenter[None,:]+radius*np.sin(TH)
    C=np.tile(sp[fields[choice]].values,(len(theta),1))
    sx,sy=np.meshgrid(x,np.linspace(-25,25,25))
    sz=-depth+.8*np.sin(sx/45)+.15*np.sin(sy/5)
    fig=go.Figure([go.Surface(x=X,y=Y,z=Z,surfacecolor=C,colorscale="Turbo",colorbar=dict(title=choice),name="Pipeline",hovertemplate="Distance %{x:.1f} m<br>Value %{surfacecolor:.3g}<extra></extra>"),go.Surface(x=sx,y=sy,z=sz,colorscale=[[0,"#514b40"],[1,"#a29477"]],showscale=False,hoverinfo="skip",name="Seabed")])
    ids=[np.argmin(abs(x-v)) for v in [0,100,180,300,400,500]]
    fig.add_trace(go.Scatter3d(x=x[ids],y=np.zeros(len(ids)),z=zcenter[ids]+radius+1,mode="markers+text",text=[f"S{j+1}" for j in range(len(ids))],marker=dict(size=5,color="cyan"),name="Virtual sensors"))
    fig.add_trace(go.Scatter3d(x=[x[i]],y=[0],z=[zcenter[i]+radius+2],mode="markers",marker=dict(size=8,color="white"),name="Selected station"))
    if animate:
        frames=[]
        for phase in np.linspace(0,2*np.pi,24,endpoint=False):
            shift=span*latest.amplitude_mm/1000*100*np.sin(phase)
            frames.append(go.Frame(data=[go.Surface(y=Y+shift[None,:])],traces=[0]))
        fig.frames=frames
        fig.update_layout(updatemenus=[dict(type="buttons",buttons=[dict(label="Play VIV",method="animate",args=[None,{"frame":{"duration":90,"redraw":True},"fromcurrent":True}]),dict(label="Pause",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False}}])])])
    fig.update_layout(template="plotly_dark",height=620,scene=dict(xaxis_title="Chainage (m)",yaxis_title="Offset (m)",zaxis_title="Elevation (m)",aspectmode="manual",aspectratio=dict(x=5,y=1,z=.65),camera=dict(eye=dict(x=1.4,y=1.8,z=.9))),margin=dict(l=0,r=0,b=0,t=20))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Synthetic spatial profiles; geometry exaggerated; VIV lateral displacement magnified 100× in animation. Sensor markers are fixed reference locations.")
    st.plotly_chart(px.line(sp,x="distance_m",y=fields[choice]),use_container_width=True)
    st.download_button("Download interactive 3D HTML",fig.to_html(include_plotlyjs=True),"FEDDO_3D.html","text/html")
def chart(cols):
    st.plotly_chart(px.line(df,x="timestamp",y=cols),use_container_width=True)
with tabs[1]:
    chart(["pressure_bar"]); chart(["temperature_c"]); chart(["hoop_mpa","thermal_mpa"])
    st.caption("Gauge pressure reference; thin-wall hoop stress using minimum model thickness. No burst/collapse assessment.")
with tabs[2]:
    st.plotly_chart(px.line(sp,x="distance_m",y="thickness_mm").add_hline(y=tmin,line_dash="dash"),use_container_width=True)
    years=np.arange(31)
    st.plotly_chart(px.line(x=years,y=np.maximum(0,thick.min()-rate[np.argmin(thick)]*years),labels={"x":"Future years","y":"Hotspot thickness (mm)"}).add_hline(y=tmin,line_dash="dash"),use_container_width=True)
with tabs[3]:
    st.metric("Estimated first natural frequency",f"{fn:.3f} Hz")
    chart(["frequency_ratio"]); chart(["amplitude_mm"]); chart(["fatigue_damage"])
    st.caption("Simply supported beam, no axial force; prescribed VIV envelope; illustrative S–N A=1e12, m=3. Damage over observed intervals only, not lifetime damage.")
with tabs[4]:
    st.caption("Isolation Forest trained on first third of history, assumed baseline. Flags are not failure probabilities.")
    st.plotly_chart(px.scatter(df,x="timestamp",y="pressure_bar",color="ai_anomaly"),use_container_width=True)
    chart(["ai_score"])
    st.dataframe(df[df.ai_anomaly],hide_index=True)
with tabs[5]:
    st.dataframe(df,hide_index=True)
    st.download_button("Download sensor history",df.to_csv(index=False),"sensor_history.csv","text/csv")
    st.download_button("Download spatial model",sp.to_csv(index=False),"spatial_model.csv","text/csv")
    st.caption("CSV input: timestamp, pressure_bar, temperature_c, current_m_s. Pressure is gauge. Spatial fields remain synthetic even when CSV is imported.")
st.sidebar.button("Refresh simulation snapshot")
st.sidebar.caption("Refresh reruns the synthetic snapshot; not a continuous live sensor service. No data persistence.")
