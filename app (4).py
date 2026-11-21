import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from scipy.optimize import brentq
import streamlit as st

st.set_page_config(page_title='FEDDO GROUP | Integrated Field V7.0',layout='wide')
st.title('FEDDO GROUP — Integrated Subsea Field Digital Twin V7.0')
st.warning('Synthetic single-phase engineering demonstration. No physical control connection; not for operational or safety decisions.')
with st.sidebar:
    st.header('Field configuration')
    H=st.slider('Water depth (m)',500,3000,1500)
    X=st.slider('SCR layback (m)',400,2000,1000)
    age=st.slider('Operating age (years)',0,25,5)
    pr0=st.slider('Initial reservoir pressure (bar gauge)',250,650,450)
    depletion=st.slider('Assumed depletion (bar/year)',0,15,4)
    back=st.slider('FPSO arrival pressure (bar gauge)',10,100,30)
    current=st.slider('Current (m/s)',0.,2.,0.8)
    cr=st.slider('Corrosion rate (mm/year)',0.,0.6,0.2)
    heave=st.slider('FPSO heave amplitude (m)',0.,5.,1.5)
    surge=st.slider('FPSO surge amplitude (m)',0.,20.,3.)
    fault=st.selectbox('Fault scenario',['Normal','Hydraulic degradation','Well 2 isolation','Annulus pressure rise','Cooling failure (loss of insulation)','Pressure sensor bias'])
    hours=st.slider('Observation hours',48,720,240)
    seed=st.number_input('Simulation seed',0,99999,42)
    opening=[st.slider(f'Well {i+1} choke opening (%)',0,100,70,key=f'ch{i}')/100 for i in range(4)]
    if fault=='Well 2 isolation': opening[1]=0
D=.324; t0=20.6; tmin=14.; E=207e9; rho=850.; rw=1025.; g=9.81
pr=pr0-age*depletion; hydro=rho*g*H/1e5
# Common manifold pressure solved by mass balance. All pressures atmospheric gauge.
# Linear productivity includes assumed well hydrostatic head and well friction in effective PI.
head= rho*g*1800/1e5
J=np.array([18.,22.,16.,20.]); pres=pr+np.array([0.,-12.,8.,-5.]); supply=pres-head
R=.0000025

def rates(pm): return np.maximum(0,supply-pm)*J*np.array(opening)
def export(pm): return np.sqrt(max(0,pm-back-hydro)/R)
base=back+hydro
upper=max(base+1,float(supply.max())+1)
pm=brentq(lambda p:rates(p).sum()-export(p),base,upper) if rates(base).sum()>0 else base
q=rates(pm); total=q.sum()
# Ideal catenary with horizontal touchdown tangent.
a=X/brentq(lambda u:(np.cosh(u)-1)/u-H/X,1e-7,20)
x=np.linspace(0,X,181); z=a*(np.cosh(x/a)-1)-H; s=a*np.sinh(x/a); L=s[-1]
t=t0-cr*age*(1+.5*np.exp(-(s/100)**2)); life=np.maximum(0,t-tmin)/(cr*(1+.5*np.exp(-(s/100)**2))) if cr else np.full_like(s,np.inf)
di=D-2*t/1000; As=np.pi*(D*D-di*di)/4; Ai=np.pi*di*di/4; Ao=np.pi*D*D/4
w=g*(7850*As+rho*Ai-rw*Ao); tension=w.mean()*a*np.cosh(x/a)/1000
pressure=pm-rho*g*(z+H)/1e5-R*total**2*s/L
# Export friction distributed along suspended riser; no separate flowline hydraulic model.
Tin=65.; decay=400 if fault=='Cooling failure (loss of insulation)' else 1800
T=4+(Tin-4)*np.exp(-s/decay)
pe=rw*g*(-z)/1e5; hoop=(pressure-pe)*1e5*D/(2*t/1000)/1e6
mass=7850*As+rho*Ai+rw*Ao
fn=np.sqrt(np.maximum(tension,0)*1000/mass)/100
fs=.2*current/D; ratio=fs/np.maximum(fn,1e-9); amp=10*np.exp(-((ratio-1)/.3)**2)
stress=2*E*(D/2)*(np.pi/50)**2*amp/1000/1e6*1.5
fatigue=hours*3600*fs*stress**3/1e12
sp=pd.DataFrame(dict(chainage_m=s,elevation_m=z,pressure_bar=pressure,temperature_c=T,wall_mm=t,tension_kN=tension,hoop_MPa=hoop,VIV_mm=amp,fatigue_damage=fatigue,corrosion_life_y=life))
well=pd.DataFrame({'Well':[f'W{i+1:02}' for i in range(4)],'Reservoir pressure bar':pres,'Available tree pressure bar':supply,'Choke opening %':np.array(opening)*100,'Liquid rate m3/day':q})
rng=np.random.default_rng(int(seed)); h=np.arange(hours); wave=np.sin(h/8)
df=pd.DataFrame({'timestamp':pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h'),periods=hours,freq='h'),'pressure_bar':pm+wave+rng.normal(0,.3,hours),'temperature_c':Tin+wave+rng.normal(0,.2,hours),'hydraulic_bar':210+rng.normal(0,2,hours),'annulus_bar':20+rng.normal(0,.5,hours),'current_m_s':np.maximum(0,current+rng.normal(0,.03,hours)),'voltage_v':24+rng.normal(0,.1,hours)})
if fault=='Hydraulic degradation': df.loc[hours-12:,'hydraulic_bar']-=115
if fault=='Annulus pressure rise': df.loc[hours-12:,'annulus_bar']+=55
if fault=='Pressure sensor bias': df.loc[hours-12:,'pressure_bar']+=60
if fault=='Cooling failure (loss of insulation)': df.loc[hours-12:,'temperature_c']-=15
features=list(df.columns[1:]); model=IsolationForest(n_estimators=80,contamination=.03,random_state=int(seed)).fit(df[features].iloc[:hours//2]); df['anomaly']=model.predict(df[features])==-1; df['score']=model.decision_function(df[features])
for c,(label,value) in zip(st.columns(5),[('Field liquid rate',f'{total:,.0f} m³/day'),('Manifold pressure',f'{pm:.1f} bar'),('Top tension',f'{tension[-1]:,.0f} kN'),('Min SCR wall',f'{t.min():.2f} mm'),('FPSO arrival temp',f'{T[-1]:.1f} °C')]): c.metric(label,value)
flags=[]
if total==0: flags.append('No forward production in simplified network')
if t.min()<=tmin: flags.append('Wall thickness screening limit reached')
if T.min()<15: flags.append('Low temperature screening; hydrate equilibrium not calculated')
if df.hydraulic_bar.iloc[-1]<150: flags.append('Low hydraulic supply')
if df.annulus_bar.iloc[-1]>50: flags.append('High annulus pressure')
if df.anomaly.iloc[-1]: flags.append('AI anomaly')
if np.any(pressure<pe): flags.append('Net external pressure; collapse not assessed')
if flags: st.warning(' | '.join(flags))
tabs=st.tabs(['3D integrated field','Reservoir & wells','Production network','SCR & FPSO','Integrity & flow assurance','Controls & AI','Export & assumptions'])
with tabs[0]:
    field=st.selectbox('SCR colour contour',list(sp.columns[2:]))
    station=st.slider('SCR inspection position (%)',0,100,50); idx=round(station/100*(len(s)-1)); st.dataframe(sp.iloc[[idx]],hide_index=True)
    fig=go.Figure()
    def line(xx,yy,zz,name,color,width=7): fig.add_trace(go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',name=name,line=dict(color=color,width=width)))
    def box(cx,cy,cz,dx,dy,dz,name,color):
        xx=np.array([-1,1,1,-1,-1,1,1,-1])*dx/2+cx; yy=np.array([-1,-1,1,1,-1,-1,1,1])*dy/2+cy; zz=np.array([-1,-1,-1,-1,1,1,1,1])*dz/2+cz
        fig.add_trace(go.Mesh3d(x=xx,y=yy,z=zz,i=[0,0,4,4,0,0,1,1,2,2,3,3],j=[1,2,5,6,1,5,2,6,3,7,0,4],k=[2,3,6,7,5,4,6,5,7,6,4,7],color=color,name=name,showlegend=True,flatshading=True)); return len(fig.data)-1
    bx,by=np.meshgrid(np.linspace(-1100,X+250,40),np.linspace(-650,650,30)); fig.add_trace(go.Surface(x=bx,y=by,z=np.full_like(bx,-H),colorscale=[[0,'#665b44'],[1,'#665b44']],showscale=False,name='Seabed',opacity=.85))
    for i,(wx,wy) in enumerate([(-900,-350),(-900,350),(-550,-350),(-550,350)]):
        box(wx,wy,-H+16,30,30,32,f'XT {i+1}', '#d8aa24' if q[i]>0 else '#a54545'); line([wx,wx],[wy,wy],[-H,-H-160],f'Well {i+1} (schematic depth)','#999999',4)
        line([wx,-250,-250],[wy,wy,0],[-H+8]*3,f'Jumper {i+1}','#e5b52f'); line([wx,-200,X],[wy,wy,0],[-H+12,-H+12,0],f'Control route {i+1}','#31b8c7',2)
    box(-250,0,-H+15,65,65,30,'Production manifold','#d3a524'); line([-250,0],[0,0],[-H+5,-H],'Export flowline','#f1b33d')
    color=np.asarray(sp[field]); color=np.where(np.isfinite(color),color,np.nan); color=np.nan_to_num(color)
    fig.add_trace(go.Scatter3d(x=x,y=np.zeros_like(x),z=z,mode='lines',name='SCR',line=dict(color=color,colorscale='Turbo',width=10,colorbar=dict(title=field),showscale=True),customdata=s,hovertemplate='Chainage %{customdata:.1f} m<extra>SCR</extra>')); riser=len(fig.data)-1
    hull=box(X+70,0,8,180,65,22,'FPSO hull','#b43d38'); deck=box(X+60,0,29,100,48,20,'FPSO topsides','#c1ced4'); line([X,X],[0,0],[0,8],'Hang-off','#eeeeee',5)
    fig.add_trace(go.Scatter3d(x=[x[idx]],y=[0],z=[z[idx]],mode='markers',marker=dict(size=6,color='white'),name='Inspection station'))
    if st.checkbox('Enable animated vessel & SCR movement',True):
        frames=[]; hullx=np.array(fig.data[hull].x); hullz=np.array(fig.data[hull].z); deckx=np.array(fig.data[deck].x); deckz=np.array(fig.data[deck].z)
        # Move hull, topsides, SCR and hang-off together; prescribed motion only.
        for k,ph in enumerate(np.linspace(0,2*np.pi,30,endpoint=False)):
            dx=surge*np.sin(ph); dz=heave*np.cos(ph); f=s/L
            frames.append(go.Frame(name=str(k),traces=[riser,hull,deck,deck+1],data=[go.Scatter3d(x=x+dx*f,y=amp/1000*30*np.sin(ph+f*8)*np.sin(np.pi*f),z=z+dz*f),go.Mesh3d(x=hullx+dx,z=hullz+dz),go.Mesh3d(x=deckx+dx,z=deckz+dz),go.Scatter3d(x=[X+dx,X+dx],z=[dz,8+dz])]))
        fig.frames=frames; fig.update_layout(updatemenus=[dict(type='buttons',buttons=[dict(label='Play motion',method='animate',args=[None,{'frame':{'duration':100,'redraw':True},'fromcurrent':True}]),dict(label='Pause',method='animate',args=[[None],{'mode':'immediate','frame':{'duration':0}}])])])
    fig.update_layout(template='plotly_dark',height=720,scene=dict(aspectmode='data',xaxis_title='East (m)',yaxis_title='North (m)',zaxis_title='Elevation (m)'),margin=dict(l=0,r=0,t=20,b=0)); st.plotly_chart(fig,use_container_width=True)
    st.caption('Schematic asset geometry; well depth exaggerated/truncated; VIV visual displacement multiplied by 30. Motion is prescribed, not a solved dynamic response.')
    st.download_button('Export interactive 3D HTML',fig.to_html(include_plotlyjs=True),'FEDDO_Field_V7.html','text/html')
with tabs[1]:
    st.dataframe(well,hide_index=True); st.plotly_chart(px.bar(well,x='Well',y='Liquid rate m3/day'),use_container_width=True)
    st.caption('Assumed linear depletion and productivity with 1,800 m liquid head; no reservoir material balance or multiphase well model.')
with tabs[2]:
    st.metric('Network mass balance residual',f'{total-export(pm):.6f} m³/day'); st.plotly_chart(px.line(sp,x='chainage_m',y='pressure_bar'),use_container_width=True); st.caption('Common manifold pressure solved by mass balance; quadratic export friction; no transient or multiphase network hydraulics.')
with tabs[3]:
    st.metric('Suspended SCR length',f'{L:.1f} m'); st.metric('Catenary parameter',f'{a:.1f} m')
    for col in ['tension_kN','VIV_mm','fatigue_damage']: st.plotly_chart(px.line(sp,x='chainage_m',y=col),use_container_width=True)
    tm=np.linspace(0,60,300); motion=pd.DataFrame({'seconds':tm,'heave_m':heave*np.cos(2*np.pi*tm/12),'surge_m':surge*np.sin(2*np.pi*tm/12)}); st.plotly_chart(px.line(motion,x='seconds',y=['heave_m','surge_m']),use_container_width=True)
    st.caption('Ideal catenary, local 50 m tension-dominated frequency proxy, prescribed VIV amplitude, illustrative S–N curve A=1e12 and m=3; observed-period estimate only.')
with tabs[4]:
    for col in ['wall_mm','corrosion_life_y','hoop_MPa','temperature_c']: st.plotly_chart(px.line(sp.replace([np.inf,-np.inf],np.nan),x='chainage_m',y=col),use_container_width=True)
    st.caption('15°C is an illustrative low-temperature flag, not a hydrate prediction. No wax, slugging, collapse, burst or fracture assessment.')
with tabs[5]:
    for col in features: st.plotly_chart(px.line(df,x='timestamp',y=col),use_container_width=True)
    st.plotly_chart(px.scatter(df,x='timestamp',y='score',color='anomaly'),use_container_width=True); st.dataframe(df[df.anomaly],hide_index=True); st.caption('AI trained on first half of synthetic history; anomaly flags are not failure probabilities. No physical commands.')
with tabs[6]:
    for name,data in [('wells',well),('SCR_spatial',sp),('sensor_history',df)]: st.download_button(f'Download {name} CSV',data.to_csv(index=False),f'{name}.csv','text/csv')
    st.markdown('**Limitations:** independent synthetic control signals; no real-time connections or persistent history; simplified single-phase steady-state production; fixed touchdown and no soil interaction; no validated fatigue or safety assessment. DNV-ST-F201, DNV-ST-F101, DNV-RP-C203 and API 17-series are future validation references, not claims of compliance.')
