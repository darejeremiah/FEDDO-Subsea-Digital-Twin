import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
st.set_page_config(page_title='FEDDO GROUP | Subsea Hardware V6.0',layout='wide')
st.title('FEDDO GROUP — Subsea Production Hardware Digital Twin V6.0')
st.warning('Synthetic engineering prototype. No physical control connection, certified integrity assessment or operational commands.')
with st.sidebar:
 st.header('Simulation configuration')
 depth=st.number_input('Water depth (m)',100,4000,1500)
 p=st.slider('Tree inlet pressure (bar gauge)',50,500,250)
 back=st.slider('Manifold backpressure (bar gauge)',10,300,100)
 temp=st.slider('Inlet temperature (°C)',10,150,70)
 opening=st.slider('Choke opening (%)',0,100,60)
 master=st.checkbox('Production master valve open',True)
 wing=st.checkbox('Production wing valve open',True)
 hyd=st.slider('Hydraulic supply differential (bar)',0,400,210)
 age=st.slider('Asset age (years)',0,30,5)
 rate=st.slider('Corrosion rate (mm/year)',0.,1.,.2)
 fault=st.selectbox('Inject simulated fault',['None','Pressure excursion','Hydraulic degradation','Annulus pressure rise','Leak indicator','Slow valve response','Electrical degradation'])
 seed=st.number_input('Random seed',0,9999,42)
n=240; rng=np.random.default_rng(seed); h=np.arange(n); mask=h>=220
P=p+2*np.sin(h/13)+rng.normal(0,.5,n); T=temp+rng.normal(0,.6,n); H=hyd+rng.normal(0,1,n); A=20+rng.normal(0,.6,n); leak=np.maximum(0,rng.normal(.02,.01,n)); volts=24+rng.normal(0,.15,n); response=5*210/np.maximum(H,1)+rng.normal(0,.2,n)
if fault=='Pressure excursion': P[mask]+=60
if fault=='Hydraulic degradation': H[mask]*=.45
if fault=='Annulus pressure rise': A[mask]+=55
if fault=='Leak indicator': leak[mask]+=.8
if fault=='Slow valve response': response[mask]+=15
if fault=='Electrical degradation': volts[mask]-=8
# Illustrative liquid orifice only, not a multiphase choke correlation.
rho=850; area=np.pi*(.025**2)/4*opening/100
Q=.62*area*np.sqrt(2*np.maximum(P-back,0)*1e5/rho)*86400*int(master and wing)
external=1025*9.81*depth/1e5
thickness=max(0,25-rate*age); life=max(0,(thickness-15)/rate) if rate else np.inf
D=.18; hoop=(P-external)*1e5*D/(2*thickness/1000)/1e6 if thickness>0 else np.full(n,np.nan)
df=pd.DataFrame(dict(timestamp=pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h'),periods=n,freq='h'),pressure_bar=P,temperature_c=T,hydraulic_bar=H,annulus_bar=A,leak_indicator=leak,voltage_v=volts,response_s=response,flow_m3_day=Q,hoop_mpa=hoop))
features=['pressure_bar','temperature_c','hydraulic_bar','annulus_bar','leak_indicator','voltage_v','response_s']
upload=st.sidebar.file_uploader('Optional sensor CSV',type='csv')
if upload:
 try:
  d=pd.read_csv(upload); d=d[['timestamp']+features].copy(); d.timestamp=pd.to_datetime(d.timestamp,utc=True,errors='raise'); d[features]=d[features].apply(pd.to_numeric,errors='raise'); d=d.sort_values('timestamp').reset_index(drop=True)
  if len(d)<48 or d.timestamp.duplicated().any() or not np.isfinite(d[features]).all().all(): raise ValueError('Require 48+ unique finite samples')
  df=d; df['flow_m3_day']=.62*area*np.sqrt(2*np.maximum(df.pressure_bar-back,0)*1e5/rho)*86400*int(master and wing); df['hoop_mpa']=(df.pressure_bar-external)*1e5*D/(2*thickness/1000)/1e6 if thickness>0 else np.nan
 except Exception as e: st.error(str(e)); st.stop()
model=IsolationForest(random_state=seed,contamination=.03).fit(df[features].iloc[:len(df)//3]); df['anomaly']=model.predict(df[features])==-1; df['anomaly_score']=model.decision_function(df[features]); r=df.iloc[-1]
alerts={'Tree pressure':r.pressure_bar>350,'Annulus':r.annulus_bar>50,'Hydraulics':r.hydraulic_bar<150,'Electrical':r.voltage_v<20,'Leak indicator':r.leak_indicator>.2,'Valve response':r.response_s>12,'Wall thickness':thickness<=15,'Reverse differential':r.pressure_bar<external}
for c,(k,v) in zip(st.columns(5),[('Tree pressure',f'{r.pressure_bar:.1f} bar'),('Flow estimate',f'{r.flow_m3_day:.0f} m³/day'),('Annulus pressure',f'{r.annulus_bar:.1f} bar'),('Hydraulic supply',f'{r.hydraulic_bar:.1f} bar'),('AI anomaly','Yes' if r.anomaly else 'No')]): c.metric(k,v)
st.write('Illustrative screening flags: '+(', '.join(k for k,v in alerts.items() if v) or 'None in latest sample'))
tabs=st.tabs(['3D virtual hardware','Production & choke','Valves & annulus','Control systems','Integrity','AI & alerts','Export & assumptions'])
with tabs[0]:
 mode=st.selectbox('Equipment display',['Equipment identification','Screening status']); station=st.selectbox('Inspect component',['Wellhead','Master valve','Wing valve','Choke','Annulus valve','SCM','Manifold'])
 fig=go.Figure()
 def cylinder(name,a,b,rad,color,flag=False):
  a=np.array(a,float); b=np.array(b,float); axis=b-a; axis/=np.linalg.norm(axis); ref=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0.,1,0]); u=np.cross(axis,ref); u/=np.linalg.norm(u); v=np.cross(axis,u); th=np.linspace(0,2*np.pi,32); s=np.linspace(0,1,8); pts=a[None,None,:]+s[:,None,None]*(b-a)[None,None,:]+rad*(np.cos(th)[None,:,None]*u+np.sin(th)[None,:,None]*v); col=('#ef4444' if flag else '#36a878') if mode=='Screening status' else color; fig.add_trace(go.Surface(x=pts[:,:,0],y=pts[:,:,1],z=pts[:,:,2],surfacecolor=np.zeros(pts.shape[:2]),colorscale=[[0,col],[1,col]],showscale=False,name=name,hovertemplate=name+'<extra></extra>'))
 def pipe(name,pts,color):
  q=np.array(pts); fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode='lines',line=dict(width=10,color=color),name=name))
 X,Y=np.meshgrid(np.linspace(-7,24,30),np.linspace(-8,8,20)); fig.add_trace(go.Surface(x=X,y=Y,z=np.zeros_like(X),colorscale=[[0,'#625b47'],[1,'#82775a']],showscale=False,hoverinfo='skip'))
 for x in [0,8]:
  cylinder('Wellhead',(x,0,0),(x,0,1.4),.65,'#777c80',alerts['Wall thickness']); cylinder('Tree body',(x,0,1.4),(x,0,3.9),.65,'#e5b52b',alerts['Tree pressure']); cylinder('Master valve',(x,0,2),(x,0,2.7),.85,'#d9a315',alerts['Valve response']); cylinder('Tree cap',(x,0,3.9),(x,0,5),.42,'#72777b'); cylinder('Wing valve',(x,0,3),(x,2,3),.45,'#e5b52b',alerts['Valve response']); cylinder('Choke',(x,2,2.4),(x,2,3.7),.48,'#db8b21',alerts['Tree pressure']); cylinder('Annulus valve',(x,0,2),(x,-1.8,2),.35,'#d9a315',alerts['Annulus']); cylinder('SCM',(x-1.3,-1,0.5),(x-1.3,-1,2.2),.55,'#368bb5',alerts['Hydraulics'] or alerts['Electrical']); pipe('Production jumper',[(x,2,3),(x,3,1),(x,5,.5),(18,5,.5),(18,1,1)],'#e6bc40'); pipe('Protection frame',[(x-2,-2,.2),(x+2,-2,.2),(x+2,2,.2),(x-2,2,.2),(x-2,-2,.2)],'#dbb34c')
 cylinder('Manifold header',(16,0,1),(22,0,1),.55,'#e5b52b',alerts['Leak indicator']); pipe('Export flowline',[(22,0,1),(24,0,.3)],'#8799a7'); pipe('Control umbilical',[(0,-1,1),(0,-5,.2),(8,-5,.2),(8,-1,1),(18,-5,.2)],'#41b6cf')
 fig.update_layout(template='plotly_dark',height=650,scene=dict(aspectmode='data',xaxis_title='Field X (m)',yaxis_title='Y (m)',zaxis_title='Height (m)'),margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig,use_container_width=True)
 metrics={'Wellhead':f'Wall thickness {thickness:.2f} mm','Master valve':f'Commanded position: {master}; response {r.response_s:.1f} s','Wing valve':f'Commanded position: {wing}','Choke':f'Opening {opening}%; flow {r.flow_m3_day:.0f} m³/day','Annulus valve':f'Annulus {r.annulus_bar:.1f} bar','SCM':f'Hydraulics {r.hydraulic_bar:.1f} bar; supply {r.voltage_v:.1f} V','Manifold':f'Assumed backpressure {back} bar'}; st.info(metrics[station]); st.caption('Schematic two-well geometry; both trees use shared synthetic readings. Colours indicate selected screening flags, not verified equipment health.'); st.download_button('Export interactive 3D HTML',fig.to_html(include_plotlyjs=True),'FEDDO_hardware.html','text/html')
def chart(c): st.plotly_chart(px.line(df,x='timestamp',y=c),use_container_width=True)
with tabs[1]:
 chart('pressure_bar'); chart('temperature_c'); chart('flow_m3_day'); st.caption('Single-phase incompressible orifice approximation; no multiphase or critical-flow model. Flow is zero when either commanded isolation valve is closed.')
with tabs[2]:
 chart('annulus_bar'); chart('response_s'); st.write('Master / wing commanded state:',master,wing); st.caption('Commanded states are simulated, not measured valve feedback.')
with tabs[3]:
 chart('hydraulic_bar'); chart('voltage_v'); chart('leak_indicator')
with tabs[4]:
 st.metric('Remaining wall thickness',f'{thickness:.2f} mm'); st.metric('Corrosion screening life',f'{life:.1f} years'); chart('hoop_mpa'); years=np.arange(31); st.plotly_chart(px.line(x=years,y=np.maximum(0,thickness-rate*years),labels={'x':'Future years','y':'Thickness (mm)'}).add_hline(y=15),use_container_width=True); st.caption('Uniform corrosion and thin-wall tubular screening only; not a wellhead or valve-body stress analysis. Collapse, seals, erosion and fatigue not solved.')
with tabs[5]:
 st.plotly_chart(px.scatter(df,x='timestamp',y='pressure_bar',color='anomaly'),use_container_width=True); chart('anomaly_score'); st.dataframe(df[df.anomaly]); st.caption('Isolation Forest uses first third as assumed normal baseline; anomaly is not a failure diagnosis.')
with tabs[6]:
 st.dataframe(df); st.download_button('Export monitoring CSV',df.to_csv(index=False),'FEDDO_hardware_history.csv','text/csv'); st.write('CSV input columns: timestamp, '+', '.join(features)); st.write('All thresholds illustrative. Pressure gauge relative to atmosphere, hydraulic supply differential relative to ambient. No live control, persistent storage, verified CAD, seal model or safety certification.')
