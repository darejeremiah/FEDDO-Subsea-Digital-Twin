import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import brentq
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title='FEDDO GROUP | FPSO SCR Digital Twin',layout='wide')
st.title('FEDDO GROUP | FPSO–SCR Digital Twin V3.0')
st.caption('Synthetic engineering screening prototype — not a coupled hydrodynamic solver or certified integrity assessment.')
with st.sidebar:
    st.header('Asset and environment')
    H=st.number_input('Water depth (m)',100.,3000.,1000.)
    R=st.number_input('Horizontal layback (m)',50.,5000.,750.)
    D=st.number_input('Outside diameter (m)',.15,.8,.324,format='%.3f')
    t0=st.number_input('Nominal wall (mm)',5.,50.,20.6)
    tm=st.number_input('Minimum screening wall (mm)',2.,40.,14.)
    age=st.slider('Operating age (years)',0,30,5)
    cr=st.slider('Base corrosion rate (mm/year)',0.,1.,.2)
    P=st.slider('Top internal gauge pressure (bar)',0,400,135)
    T=st.slider('Top fluid temperature (°C)',4,150,65)
    U=st.slider('Surface current (m/s)',0.,3.,.8)
    heave=st.slider('Heave amplitude (m)',0.,8.,1.5)
    surge=st.slider('Surge amplitude (m)',0.,20.,3.)
    sway=st.slider('Sway amplitude (m)',0.,20.,2.)
    roll=st.slider('Roll amplitude (degrees)',0.,10.,2.)
    pitch=st.slider('Pitch amplitude (degrees)',0.,10.,1.)
    yaw=st.slider('Yaw amplitude (degrees)',0.,10.,1.)
    period=st.slider('Motion period (s)',5.,25.,12.)
    exposure=st.slider('Assumed sea-state exposure (hours)',1,8760,240)
    viv=st.slider('Peak VIV amplitude / diameter',0.,1.,.15)
    scf=st.slider('Assumed fatigue SCF',1.,4.,1.5)
    inject=st.checkbox('Inject synthetic pressure anomaly',True)
if tm>=t0 or 2*t0/1000>=D:
    st.error('Invalid wall geometry or minimum thickness.'); st.stop()
E=207e9; rho=1025.; rhoi=850.; g=9.81
# Ideal flexible, inextensible catenary tangent to a horizontal seabed.
def solve_a(h,r):
    return r/brentq(lambda q:(np.cosh(q)-1)/q-h/r,1e-7,30)
a=solve_a(H,R)
x=np.linspace(0,R,241); z=a*(np.cosh(x/a)-1)-H
s=a*np.sinh(x/a); length=s[-1]; f=s/length
rate=cr*(1+1.5*np.exp(-(s/(.09*length))**2)); t=t0-age*rate
if min(t)<=0:
    st.error('Through-wall loss predicted. Model stopped.'); st.stop()
di=D-2*t/1000; A=np.pi*(D**2-di**2)/4; I=np.pi*(D**4-di**4)/64
mass=7850*A+rhoi*np.pi*di**2/4
w=(mass-rho*np.pi*D**2/4)*g
if min(w)<=0:
    st.error('Non-positive submerged weight; simple hanging catenary not applicable.'); st.stop()
w0=float(np.mean(w)); tension=w0*a*np.cosh(x/a); curvature=1/(a*np.cosh(x/a)**2)
pe=rho*g*(-z)/1e5; pi=P+rhoi*g*(-z)/1e5; dp=pi-pe
hoop=dp*1e5*D/(2*t/1000)/1e6
thermal=4+(T-4)*np.exp(-(length-s)/1500)
bend=E*D/2*curvature/1e6
current=U*(.25+.75*(1+z/H))
m_eff=mass+rho*np.pi*D**2/4
# Local tension-string frequency proxy with prescribed correlation length.
Lc=50.; fn=np.sqrt(tension/m_eff)/(2*Lc); fs=.2*current/D
ratio=fs/fn; av=viv*D*np.exp(-((ratio-1)/.3)**2)
# Assumed local sinusoidal VIV shape, plus quasi-static cyclic curvature proxy.
sv=2*E*D/2*(np.pi/Lc)**2*av*scf/1e6
# FPSO rotations mapped to illustrative hang-off lever arm offsets, not RAOs.
lever=20.; dx=surge+lever*np.deg2rad(pitch); dy=sway+lever*np.deg2rad(yaw); dz=heave+lever*np.deg2rad(roll)
# Fixed touchdown assumption; prescribed displacement shape, not equilibrium solution.
shape=f**2; motion_range=2*E*D/2*2*np.sqrt(dy**2+dz**2+dx**2)/length**2*scf/1e6
motion_range=motion_range*(1+4*np.exp(-(f/.08)**2))
# Illustrative single-slope S-N law, stress in MPa; no design fatigue factor.
fat=exposure*3600*(fs*sv**3+motion_range**3/period)/1e12
life=np.divide(np.maximum(t-tm,0),rate,out=np.full_like(t,np.inf),where=rate>0); life[t<=tm]=0
sp=pd.DataFrame(dict(arc_m=s,x_m=x,elevation_m=z,wall_mm=t,metal_loss_mm=t0-t,life_years=life,tension_kN=tension/1000,curvature_1_m=curvature,bending_MPa=bend,pressure_bar=pi,external_bar=pe,hoop_MPa=hoop,temperature_C=thermal,thermal_stress_MPa=-.5*E*12e-6*(thermal-4)/1e6,viv_mm=av*1000,frequency_ratio=ratio,fatigue_damage=fat))
for c,(label,value) in zip(st.columns(5),[('Suspended length',f'{length:.0f} m'),('Top tension proxy',f'{tension[-1]/1000:.0f} kN'),('Minimum wall',f'{min(t):.2f} mm'),('Peak exposure damage',f'{max(fat):.3g}'),('Minimum corrosion life',f'{min(life):.1f} y')]): c.metric(label,value)
flags=[]
if min(t)<=tm: flags.append('Minimum wall reached')
if max(abs(hoop))>250: flags.append('Illustrative hoop stress limit exceeded')
if np.any(dp<0): flags.append('Net external pressure: collapse not assessed')
if max(fat)>=1: flags.append('Illustrative fatigue damage >= 1')
if np.any((ratio>.8)&(ratio<1.2)): flags.append('Local VIV resonance proxy flagged')
if flags: st.warning(' | '.join(flags))
else: st.info('No configured flags; not a safety determination.')
tabs=st.tabs(['3D FPSO & SCR','Static & dynamic response','VIV & fatigue','Corrosion','Pressure & thermal','AI monitoring','Export & assumptions'])
with tabs[0]:
    fields=['tension_kN','bending_MPa','curvature_1_m','wall_mm','metal_loss_mm','pressure_bar','temperature_C','viv_mm','fatigue_damage']
    field=st.selectbox('Riser colour contour',fields)
    station=st.slider('Inspection position (% arc length from touchdown)',0,100,10)
    j=np.argmin(abs(f-station/100)); st.dataframe(sp.iloc[[j]],hide_index=True)
    animation=st.checkbox('Enable animated FPSO and SCR motion')
    mag=st.slider('Motion visual magnification',1,30,5)
    def riser(phase=0, moving=False):
        xx=x.copy(); yy=np.zeros_like(x); zz=z.copy()
        if moving:
            xx+=mag*dx*np.sin(phase)*shape
            yy+=mag*(dy*np.cos(phase)*shape+av*np.sin(8*np.pi*f)*np.sin(phase*3)*(1-f)*f*4)
            zz+=mag*dz*np.sin(phase+.3)*shape
        return xx,yy,zz
    def vessel(phase=0,moving=False):
        xx,yy,zz=riser(phase,moving); X=xx[-1]; Y=yy[-1]; Z=zz[-1]
        vertices=np.array([[-90,0,-8],[90,0,-8],[115,42,-8],[-90,42,-8],[-90,0,12],[90,0,12],[115,42,12],[-90,42,12]])+np.array([X,Y,Z])
        return go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=[0,0,4,4,0,0,1,1,2,2,3,3],j=[1,2,5,6,1,5,2,6,3,7,0,4],k=[2,3,6,7,5,4,6,5,7,6,4,7],color='#c64e3b',name='FPSO hull (schematic)',opacity=1,showlegend=True)
    def trace(phase=0,moving=False):
        xx,yy,zz=riser(phase,moving)
        return go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',line=dict(width=9,color=sp[field],colorscale='Turbo',colorbar=dict(title=field),showscale=True,cmin=float(sp[field].min()),cmax=float(sp[field].max())+1e-12),customdata=sp[['arc_m',field]].values,hovertemplate='Arc %{customdata[0]:.1f} m<br>Value %{customdata[1]:.4g}<extra></extra>',name='SCR')
    bx,by=np.meshgrid(np.linspace(-300,R+200,36),np.linspace(-180,180,24))
    fig=go.Figure([trace(),vessel(),go.Surface(x=bx,y=by,z=np.full_like(bx,-H),colorscale=[[0,'#827460'],[1,'#827460']],showscale=False,name='Seabed',hoverinfo='skip'),go.Scatter3d(x=[0,-240],y=[0,0],z=[-H,-H],mode='lines+markers',line=dict(width=7,color='orange'),name='Seabed flowline'),go.Scatter3d(x=[-240,0,x[j],R],y=[0,0,0,0],z=[-H,-H,z[j],0],mode='markers+text',text=['Manifold','Touchdown','Inspection','Hang-off'],textposition='top center',marker=dict(size=5,color='cyan'),name='Reference stations')])
    if animation:
        fig.frames=[go.Frame(data=[trace(p,True),vessel(p,True)],traces=[0,1]) for p in np.linspace(0,2*np.pi,32,endpoint=False)]
        fig.update_layout(updatemenus=[dict(type='buttons',buttons=[dict(label='Play motion',method='animate',args=[None,dict(frame=dict(duration=100,redraw=True),transition=dict(duration=0),fromcurrent=True)]),dict(label='Pause',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False))])])])
    fig.update_layout(template='plotly_dark',height=720,scene=dict(xaxis_title='Horizontal (m)',yaxis_title='Crossline (m)',zaxis_title='Elevation (m)',aspectmode='data',camera=dict(eye=dict(x=1.5,y=1.6,z=.8))),margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Schematic hull translates with hang-off; rotations represented by assumed lever-arm translations, not rigid-body rotations. Fixed touchdown. Motion magnified; pipe line thickness not to scale. Animation is illustrative, not physical time.')
    st.download_button('Download interactive 3D HTML',fig.to_html(include_plotlyjs=True),'FEDDO_SCR_3D.html','text/html')
def plot(cols):
    for col in cols: st.plotly_chart(px.line(sp,x='arc_m',y=col),use_container_width=True)
with tabs[1]:
    plot(['tension_kN','curvature_1_m','bending_MPa'])
    phase=np.linspace(0,4*np.pi,240); tt=phase*period/(2*np.pi)
    motions=pd.DataFrame(dict(time_s=tt,surge_m=surge*np.sin(phase),sway_m=sway*np.cos(phase),heave_m=heave*np.sin(phase+.3),roll_deg=roll*np.sin(phase),pitch_deg=pitch*np.sin(phase),yaw_deg=yaw*np.cos(phase)))
    st.plotly_chart(px.line(motions,x='time_s',y=['surge_m','sway_m','heave_m']),use_container_width=True)
    st.plotly_chart(px.line(motions,x='time_s',y=['roll_deg','pitch_deg','yaw_deg']),use_container_width=True)
    st.caption('Static tension only; dynamic displacement and fatigue are prescribed proxies, not solved dynamic tension or touchdown migration.')
with tabs[2]:
    plot(['frequency_ratio','viv_mm','fatigue_damage'])
    st.caption('50 m assumed local correlation length, illustrative S-N A=1e12 MPa³ cycles, m=3. Damage is for selected exposure only; not lifetime or code-compliant fatigue.')
with tabs[3]:
    plot(['metal_loss_mm','wall_mm','life_years'])
with tabs[4]:
    plot(['pressure_bar','external_bar','hoop_MPa','temperature_C','thermal_stress_MPa'])
    st.caption('Static hydrostatic internal and external profiles; no friction, multiphase flow, collapse or combined-load assessment.')
with tabs[5]:
    rng=np.random.default_rng(42); n=240; time=pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h'),periods=n,freq='h')
    data=pd.DataFrame(dict(timestamp=time,pressure_bar=P+rng.normal(0,2,n),temperature_c=T+rng.normal(0,.7,n),current_m_s=np.maximum(0,U+rng.normal(0,.06,n)),heave_m=heave*np.sin(np.arange(n)/5)+rng.normal(0,.1,n)))
    if inject: data.loc[210:215,'pressure_bar']+=35
    upload=st.file_uploader('Optional AI sensor CSV',type='csv'); features=['pressure_bar','temperature_c','current_m_s','heave_m']
    if upload:
        try:
            candidate=pd.read_csv(upload)[['timestamp']+features]; candidate['timestamp']=pd.to_datetime(candidate.timestamp,utc=True); candidate[features]=candidate[features].apply(pd.to_numeric); candidate=candidate.sort_values('timestamp').reset_index(drop=True)
            if len(candidate)<48 or candidate.timestamp.duplicated().any() or candidate.timestamp.isna().any() or not np.isfinite(candidate[features]).all().all(): raise ValueError('Need 48+ valid unique samples')
            data=candidate
        except Exception as e: st.error(str(e)); st.stop()
    model=IsolationForest(n_estimators=100,contamination=.03,random_state=42).fit(data[features].iloc[:len(data)//3]); data['anomaly']=model.predict(data[features])==-1; data['score']=model.decision_function(data[features])
    st.metric('Flagged samples',int(data.anomaly.sum())); st.plotly_chart(px.scatter(data,x='timestamp',y='pressure_bar',color='anomaly'),use_container_width=True); st.plotly_chart(px.line(data,x='timestamp',y='score'),use_container_width=True); st.dataframe(data[data.anomaly]); st.download_button('Export AI history',data.to_csv(index=False),'AI_history.csv','text/csv')
    st.caption('First third assumed baseline. CSV affects AI only, not the spatial model. Anomaly scores are not failure probabilities.')
with tabs[6]:
    st.dataframe(sp,hide_index=True); st.download_button('Export SCR spatial results',sp.to_csv(index=False),'SCR_spatial.csv','text/csv'); st.download_button('Export vessel motions',motions.to_csv(index=False),'FPSO_motion.csv','text/csv')
    st.markdown('**Limitations:** ideal flexible catenary with mean submerged weight; no bending stiffness in equilibrium, soil interaction, wave loading, RAOs, coupled dynamics, installation loads, weld detail or calibrated sensor connection. Validate independently against DNV-ST-F201, DNV-RP-F204 and DNV-RP-C203 before engineering use. Free cloud hosting is not continuous monitoring.')
