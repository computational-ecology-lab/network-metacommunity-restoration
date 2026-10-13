"""Independent panels for manuscript Figure 5, using GBIF and AVONET.

Run fetch_data.py once, then this script. Public data and all intermediate
arrays are cached in data/animals.
Use --plots-only to redraw cached results. No panel letters or composite figures
are generated; final assembly is done in PowerPoint. Figures and their ZIP are
written directly to Code/metaglv-restoration-demo/output, replacing existing files.

Empirical inputs: GBIF presences (2000-2025) and AVONET species-average mass/HWI.
SDMs are exploratory presence-background models, not occupancy estimates. The
original landscape rasters have no documented provenance in the supplied README.
Growth, competition, capacity, noise and HWI-to-km conversion are illustrative
assumptions, not fitted demographic estimates. Time and abundance are model units.

Figure 5 model settings (synthetic example: blue tit, great tit and coal tit
in Snowdonia; the manuscript caption refers here for these details)
---------------------------------------------------------------------------
Model
- Network-metacommunity model equation of Figure 2 (rf.step), with r_ik = r_i
  and a_iik = -1/K_ik, so that the single-species carrying capacity is
  r_i * K_ik. K_ik is varied as a proxy for habitat condition.

Habitat patches, K_ik and distances (pipeline of Figure 3; build_network)
- Inputs: GBIF occurrence records (2000-2025); elevation, slope, NDVI and
  NDWI layers.
- K_ik = summed habitat suitability of species i over patch k.
- Least-cost distances between patches.
- K_ik = 1 outside a species' habitat cores, so the great tit population in
  H9 is maintained by immigration.

Species differences (AVONET body mass and hand-wing index; Tobias et al. 2022)
- Growth rates r_i decrease with body mass.
- Competition coefficients a_ij (the same in all patches) scale with the
  body-mass ratio.
- Mean dispersal distance scales with the hand-wing index; dispersal between
  patches decreases exponentially with least-cost distance.
- Scaling constants are illustrative values; time and abundance are in model
  units.

Stochasticity
- Temporally autocorrelated fluctuations in K_ik, together with demographic
  noise.

Scenarios (forecast)
- Disturbance: H1-H9 corridor lost at t = 5. Restoration starts at t = 20.
- R0: no restoration.
- R1: rebuild the H1-H9 corridor.
- R2a: reintroduce great tits in H9 (8.5), no habitat improvement.
- R2b: reintroduction as R2a + improve habitat condition of H9 for the great
  tit, i.e. K (great tit, H9) raised from 1 to 4.85, so that mean abundance
  over t = 25-40 matches R1.
- In the code, trajectories a0, a1, a2a and a2b correspond to R0, R1, R2a
  and R2b.

Reference
Tobias, J.A. et al. (2022) AVONET: morphological, ecological and geographical
data for all birds. Ecology Letters, 25, 581-597.
https://doi.org/10.1111/ele.13898
"""
from pathlib import Path
import json
import pickle
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.interpolate import PchipInterpolator
import habitat_network as hn
import restoration_forecast as rf

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'animals'
OUT=ROOT/'output'
SPECIES=['Cyanistes caeruleus','Parus major','Periparus ater']
COMMON=['Blue tit','Great tit','Coal tit']
COLORS=['#1f77b4','#9467bd','#2ca02c']
SEED=2026

def traits():
    rows=json.loads((DATA/'avonet_three_tits.json').read_text())
    rows={x['Species1']:x for x in rows}
    return [rows[s] for s in SPECIES]

def environment():
    # About 100 m grid; four-fold coarsening of the supplied ~30 m rasters.
    with rasterio.open(ROOT/'data/env/elevation.tif') as ref:
        h,w=int(np.ceil(ref.height/4)),int(np.ceil(ref.width/4))
        tr=ref.transform*Affine.scale(ref.width/w,ref.height/h)
        crs=ref.crs
    names=['elevation','slope','ndvi','ndwi']
    bands=[]
    for name in names:
        with rasterio.open(ROOT/'data/env'/f'{name}.tif') as src:
            arr=np.full((h,w),np.nan,dtype='float32')
            reproject(rasterio.band(src,1),arr,src_transform=src.transform,src_crs=src.crs,
                      dst_transform=tr,dst_crs=crs,src_nodata=src.nodata,
                      dst_nodata=np.nan,resampling=Resampling.average)
            bands.append(arr)
    data=np.stack(bands)
    return dict(data=data,names=names,nodata=~np.isfinite(data).all(axis=0),
                transform=(tr.c,tr.a,tr.f,-tr.e))

def occurrences(species,env):
    source=json.loads((DATA/(species.replace(' ','_')+'.json')).read_text())
    d=pd.DataFrame(source['records'])
    u=d.coordinateUncertaintyInMeters
    d=d[u.isna()|(u<=1000)].copy()
    d=d.drop_duplicates(['decimalLongitude','decimalLatitude'])
    rr,cc=hn.to_rowcol(env['transform'],d.decimalLongitude,d.decimalLatitude)
    h,w=env['nodata'].shape
    ok=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
    d=d.loc[ok].copy();rr,cc=rr[ok],cc[ok]
    ok=~env['nodata'][rr,cc]
    d=d.loc[ok].copy();rr,cc=rr[ok],cc[ok]
    # Thin to one occupied raster cell; do not treat repeated sightings as abundance.
    d['row'],d['col']=rr,cc
    d=d.drop_duplicates(['row','col'])
    d.to_csv(DATA/(species.replace(' ','_')+'_clean.csv'),index=False)
    return d,dict(raw=source['count'],retained=len(d),
                  uncertainty_unknown=int(d.coordinateUncertaintyInMeters.isna().sum()))

def build_network():
    env=environment();h,w=env['nodata'].shape;tr=env['transform']
    suit,patches,counts_meta={},{},{}
    for s in SPECIES:
        d,meta=occurrences(s,env);counts_meta[s]=meta
        suit[s]=hn.suitability_map(env,d.decimalLongitude.to_numpy(),d.decimalLatitude.to_numpy(),seed=SEED)
        counts=np.zeros((h,w));np.add.at(counts,(d.row.to_numpy(),d.col.to_numpy()),1.)
        patches[s]=hn.habitat_patches(suit[s],gaussian_filter(counts,sigma=2.),min_px=4,max_patches=8)
        print(s,meta,'patches',patches[s].max(),flush=True)
    merged,contained,n_px=hn.merge_habitats(patches,min_px=4)
    combined=np.fmax.reduce(list(suit.values()))
    R=hn.resistance_surface(combined,env['data'][1],env['nodata'])
    # Nodata is an impassable barrier, rather than a costly shortcut.
    R[env['nodata']]=np.inf
    lat_c=tr[2]-h*tr[3]/2
    hids,seeds,dist_m,paths=hn.least_cost_distances(R,merged,combined,tr,lat_c)
    # Original code used a sum over ~30 m pixels. Use area-equivalent units so
    # arbitrary capacity units do not change when the raster grid is coarsened.
    area_factor=np.prod(hn.pixel_size_m(tr,lat_c))/540.0
    K=np.array([[np.nansum(suit[s][merged==hid])*area_factor if s in contained[hid] else 1.
                 for s in SPECIES] for hid in hids])
    tv=traits();mass=np.array([t['Mass'] for t in tv]);hwi=np.array([t['Hand-Wing.Index'] for t in tv])
    # HWI supplies relative dispersal; 2.5 km is an explicit illustrative scale,
    # not an empirical natal-dispersal distance (Sheard 2020; Weeks 2022).
    alpha=2.5*hwi/hwi[0]
    D=.1*np.exp(-dist_m[None]/1000/alpha[:,None,None])
    for d in D:np.fill_diagonal(d,0)
    net=dict(env=env,transform=tr,merged=merged,contained=contained,seeds=seeds,
             paths=paths,hids=hids,dist_km=dist_m/1000,K=K,D=D,suit=suit,
             combined_suit=combined,counts=counts_meta,traits=tv,alpha=alpha)
    with (DATA/'network.pkl').open('wb') as f:pickle.dump(net,f)
    print('Habitats:',len(hids),'capacity:',K.tolist(),flush=True)
    return net

def forecast(net):
    K,D=net['K'],net['D'];P,S=K.shape
    mass=np.array([t['Mass'] for t in net['traits']])
    rf.R=.8*(mass/11.1)**(-.25)
    # The same illustrative competition scale as the supplied demonstration,
    # with a mass-ratio prior following the supplementary CL example.
    overlap=np.full((S,S),.15);np.fill_diagonal(overlap,0.)
    comp=overlap*(mass[None,:]/mass[:,None])**.75
    A=-comp*rf.R[:,None]/K[K>1].mean()
    # Select BEFORE simulating: the largest potential immigrant input into a
    # patch outside that species' selected high-suitability cores. This is an
    # illustrative scenario choice, not evidence of observed absence or loss.
    candidates=[]
    for s in range(S):
        for dst in np.where(K[:,s]==1)[0]:
            for src in np.where(K[:,s]>1)[0]:
                candidates.append((D[s,dst,src]*rf.R[s]*K[src,s],s,src,dst))
    _,focal,src,dst=max(candidates)
    cut=D.copy();cut[:,src,dst]=0.;cut[:,dst,src]=0.
    rng=np.random.default_rng(SEED);n=40
    z=np.zeros((n,P,S));z[0]=rng.standard_normal((P,S))
    e=rng.standard_normal((n,P,S))
    for t in range(1,n):z[t]=.7*z[t-1]+np.sqrt(1-.7**2)*e[t]
    env=np.exp(.15*z-.15**2/2)
    demo=np.exp(.05*rng.standard_normal((n,P,S))-.05**2/2)
    initial=np.where(K>1,rng.uniform(.2,.6,(P,S))*K,0.)
    def run(action,Kstar=None,pulse=0.):
        N=initial.copy();traj=[N.copy()];Ka=K.copy()
        if Kstar is not None:Ka[dst,focal]=Kstar
        for t in range(n):
            if t==20 and pulse:N[dst,focal]+=pulse
            open_now=action=='kept' or t<5 or (action=='corridor' and t>=20)
            N=rf.step(N,(Ka if t>=20 else K)*env[t],D if open_now else cut,A,t,t+1)*demo[t]
            if not np.isfinite(N).all():raise ValueError('Nonfinite trajectory')
            traj.append(N.copy())
        return np.array(traj)[:,dst,focal]
    kept,a0,a1=run('kept'),run('lost'),run('corridor')
    target=a1[25:].mean();pulse=max(target-a0[20],0.)
    # Include the identical reintroduction pulse in the calibration itself.
    lo=float(K[dst,focal]);hi=max(2*lo,2.)
    while run('reintro',hi,pulse)[25:].mean()<target:
        hi*=2
        if hi>1e6:raise ValueError('Could not bracket habitat improvement')
    for _ in range(35):
        mid=(lo+hi)/2
        if run('reintro',mid,pulse)[25:].mean()<target:lo=mid
        else:hi=mid
    Kstar=(lo+hi)/2
    a2a,a2b=run('reintro',None,pulse),run('reintro',Kstar,pulse)
    assert np.array_equal(a0[:21],a1[:21])
    assert np.array_equal(a1[:21],a2a[:21])
    assert np.array_equal(a1[:21],a2b[:21])
    assert np.array_equal(kept[:6],a0[:6])
    assert abs(a2b[25:].mean()-target)<1e-6
    assert np.allclose(D,D.transpose(0,2,1))
    res=dict(focal=focal,src=src,dst=dst,kept=kept,a0=a0,a1=a1,a2a=a2a,a2b=a2b,
             pulse=pulse,Kstar=Kstar,loss=(a0[19]-kept[19])/kept[19],target=target)
    np.savez(DATA/'trajectories.npz',**res)
    summary={k:(float(v) if isinstance(v,(np.floating,float)) else int(v)) for k,v in res.items() if np.isscalar(v)}
    summary.update(dict(species=SPECIES[focal],source_habitat=int(src+1),target_habitat=int(dst+1),
                        growth_rates=rf.R.tolist(),dispersal_scales_km=net['alpha'].tolist(),
                        occurrence_counts=net['counts'],
                        R0_mean=float(a0[25:].mean()),R2a_mean=float(a2a[25:].mean()),
                        R2b_mean=float(a2b[25:].mean())))
    (DATA/'model_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return res

def draw_map(ax,net,r):
    tr=net['transform'];merged=net['merged'];h,w=merged.shape
    extent=(tr[0],tr[0]+w*tr[1],tr[2]-h*tr[3],tr[2])
    src,dst=r['src']+1,r['dst']+1
    path=net['paths'][(src,dst)]
    # Include complete source and target cores in the map, plus some context.
    rr,cc=np.where(np.isin(merged,[src,dst]))
    xs,ys=hn.to_lonlat(tr,rr,cc)
    cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
    hx=max((xs.max()-xs.min())*.65,(ys.max()-ys.min())*.85,.022)
    hy=hx*.95
    win=(max(extent[0],cx-hx),min(extent[1],cx+hx),max(extent[2],cy-hy),min(extent[3],cy+hy))
    # Crop to the data-covered rows within this longitude window. The final
    # raster row is mostly nodata; including it produced the white strip at
    # the bottom of the map. Keep geographic coordinates and data unchanged.
    c0=max(0,int(np.floor((win[0]-tr[0])/tr[1])))
    c1=min(w,int(np.ceil((win[1]-tr[0])/tr[1])))
    full_rows=np.flatnonzero(np.isfinite(net['combined_suit'][:,c0:c1]).all(axis=1))
    if not len(full_rows):
        raise ValueError('No fully covered raster row in the selected map window')
    covered_bottom=tr[2]-(int(full_rows[-1])+1)*tr[3]
    covered_top=tr[2]-int(full_rows[0])*tr[3]
    win=(*win[:2],max(win[2],covered_bottom),min(win[3],covered_top))
    ax.imshow(net['combined_suit'],extent=extent,cmap='Greys',vmin=0,vmax=1,alpha=.38,aspect='auto',rasterized=True)
    for hid in net['hids']:
        color=('#f4a34d' if hid==src else '#afbb43' if hid==dst else '#86b394')
        arr=np.ma.masked_where(merged!=hid,np.ones(merged.shape))
        ax.imshow(arr,extent=extent,cmap=ListedColormap([color]),alpha=.86,aspect='auto',rasterized=True)
    ax.contour((merged>0).astype(float),levels=[.5],extent=extent,origin='upper',colors='.4',linewidths=.35)
    for (a,b),p in net['paths'].items():
        if a<b:
            ax.plot(*hn.to_lonlat(tr,p[:,0],p[:,1]),color='#b93332',lw=.55,alpha=.4,zorder=3)
    ax.plot(*hn.to_lonlat(tr,path[:,0],path[:,1]),color='#c62828',lw=1.9,zorder=5)
    for hid in net['hids']:
        lon,lat=hn.to_lonlat(tr,*net['seeds'][hid])
        if not (win[0]<lon<win[1] and win[2]<lat<win[3]):continue
        ax.scatter(lon,lat,s=275,fc='white',ec='#142e3c',lw=1.1,zorder=8)
        ax.text(lon,lat,str(hid),ha='center',va='center',fontsize=13,fontweight='bold',zorder=9)
        for k,s in enumerate(SPECIES):
            if s in net['contained'][hid]:
                ax.scatter(lon+(k-1)*(win[1]-win[0])*.036,lat+(win[3]-win[2])*.050,
                           s=85,c=COLORS[k],ec='white',lw=.8,zorder=9)
    handles=[Line2D([],[],ls='',marker='o',mfc=c,mec='white',ms=10,label=s) for s,c in zip(COMMON,COLORS)]
    handles.extend([
        Line2D([],[],color='#c62828',lw=1.9,label=f'H{src}–H{dst} corridor'),
        Line2D([],[],color='#b93332',lw=.55,alpha=.4,label='Other corridors'),
    ])
    ax.legend(handles=handles,loc='upper left',fontsize=13,framealpha=.94,borderpad=.45,labelspacing=.3)
    ax.set(xlim=win[:2],ylim=win[2:],xlabel='Longitude',ylabel='Latitude')
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def line(ax,t,y,**kwargs):
    tt=np.linspace(t[0],t[-1],500)
    ax.plot(tt,PchipInterpolator(t,y)(tt),**kwargs)

def pulsed_line(ax,y,pulse,**kwargs):
    # Both R2a and R2b show the identical instantaneous addition at t=20.
    line(ax,np.arange(21.),y[:21],**kwargs)
    post=np.r_[y[20]+pulse,y[21:]]
    kw={**kwargs,'label':None}
    ax.plot([20,20],[y[20],post[0]],**kw)
    line(ax,np.arange(20.,41.),post,**kw)

def draw_B(ax,r):
    t=np.arange(20.);ts=np.linspace(0,19,400)
    kept,lost=r['kept'][:20],r['a0'][:20]
    ax.fill_between(ts,PchipInterpolator(t,lost)(ts),PchipInterpolator(t,kept)(ts),color='#1f77b4',alpha=.18,label='Abundance lost')
    line(ax,t,kept,color='#1f77b4',lw=1.8,label='Corridor retained')
    line(ax,t,lost,color='#df292d',lw=1.8,label='Corridor lost')
    ymax=max(kept.max(),lost.max())*1.22
    ax.axvline(5,color='.45',ls=':',lw=1.5)
    ax.text(5.25,ymax*.97,'Corridor lost (t = 5)',va='top',fontsize=13,color='.3')
    ax.annotate('',xy=(19,lost[-1]),xytext=(19,kept[-1]),arrowprops=dict(arrowstyle='<->',lw=1.4))
    ax.text(18.7,(lost[-1]+kept[-1])/2,f"{r['loss']:+.0%}",ha='right',va='center',fontweight='bold',fontsize=18)
    ax.set(xlim=(0,20),ylim=(0,ymax),xlabel='Time (model units)',ylabel=f"{COMMON[r['focal']]} abundance (model units)")
    ax.set_xticks([0,5,10,15,20]);ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc='lower right',fontsize=13)

def restoration_legend(r):
    src,dst=r['src']+1,r['dst']+1
    return [Line2D([],[],color='black',ls='--',lw=1.5,label='R0: No restoration'),
            Line2D([],[],color='#78be74',lw=2.,label=f'R1: Rebuild H{src}–H{dst} corridor'),
            Line2D([],[],color='#579cc5',lw=2.,label=f"R2a: Reintroduce {r['pulse']:.1f} (no habitat improvement)"),
            Line2D([],[],color='#f39c44',lw=2.,label=f"R2b: Improve K(H{dst}) from 1 to {r['Kstar']:.2f} + reintroduce {r['pulse']:.1f}"),
            Patch(fc='#78be74',alpha=.18,label='Difference between rebuilding and no restoration')]

def draw_CD(ax,r,panel,show_y=True):
    t=np.arange(41.);ts=np.linspace(0,40,500)
    ax.fill_between(ts,PchipInterpolator(t,r['a0'])(ts),PchipInterpolator(t,r['a1'])(ts),color='#78be74',alpha=.18)
    line(ax,t,r['a1'],color='#78be74',lw=1.9)
    key,col=('a2a','#579cc5') if panel=='C' else ('a2b','#f39c44')
    pulsed_line(ax,r[key],r['pulse'],color=col,lw=1.9)
    # Draw the dashed baseline on top so equal R0/R2a trajectories remain legible.
    line(ax,t,r['a0'],color='black',ls='--',lw=1.1)
    ymax=max(max(r[k]) for k in ['a0','a1','a2a','a2b'])*1.25
    ymax=max(ymax,(r['a0'][20]+r['pulse'])*1.2)
    for tt,ls,txt in [(5,':','Corridor lost\n(t = 5)'),(20,'-.','Restoration\n(t = 20)')]:
        ax.axvline(tt,color='.45',ls=ls,lw=1.4,zorder=0)
        ax.text(tt+.5,ymax*.025,txt,fontsize=11,color='.3',va='bottom')
    ax.set(xlim=(0,40),ylim=(0,ymax),xlabel='Time (model units)')
    ax.set_xticks([0,20,40]);ax.yaxis.set_major_locator(MaxNLocator(5))
    if show_y:ax.set_ylabel(f"{COMMON[r['focal']]} abundance in H{r['dst']+1}")
    else:ax.tick_params(labelleft=False)

def save_figure(fig,stem,dpi=400):
    for ext in ['png','pdf','svg']:
        target=OUT/f'{stem}.{ext}'
        temporary=OUT/f'.{stem}.tmp.{ext}'
        fig.savefig(temporary,dpi=dpi,facecolor='white',bbox_inches='tight',pad_inches=.04)
        temporary.replace(target)

def figures(net,r):
    plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':15,
                         'xtick.labelsize':13,'ytick.labelsize':13,'axes.linewidth':.8,
                         'svg.fonttype':'none','pdf.fonttype':42})
    # Only four independent, unlettered plots for PowerPoint assembly.
    fig,ax=plt.subplots(figsize=(6.2,6.2));draw_map(ax,net,r);fig.tight_layout();save_figure(fig,'habitat_corridor_no_caption');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,6.2));draw_B(ax,r);fig.tight_layout();save_figure(fig,'disturbance');plt.close(fig)
    for p,stem in [('C','reintroduction'),('D','habitat_improvement')]:
        fig,ax=plt.subplots(figsize=(6.2,6.2));draw_CD(ax,r,p)
        fig.subplots_adjust(left=.16,right=.98,bottom=.13,top=.78)
        handles=restoration_legend(r);chosen=[handles[0],handles[1],handles[2 if p=='C' else 3],handles[4]]
        fig.legend(handles=chosen,loc='upper left',bbox_to_anchor=(.08,1.0),fontsize=12)
        save_figure(fig,stem);plt.close(fig)
    with ZipFile(OUT/'animal_figure_panels_updated.zip','w',ZIP_DEFLATED) as archive:
        for stem in ['habitat_corridor_no_caption','disturbance','reintroduction','habitat_improvement']:
            for ext in ['png','pdf','svg']:
                path=OUT/f'{stem}.{ext}'
                archive.write(path,path.name)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    net=build_network() if '--rebuild' in sys.argv or not (DATA/'network.pkl').exists() else pickle.loads((DATA/'network.pkl').read_bytes())
    print('Network ready',flush=True)
    if '--plots-only' in sys.argv:
        with np.load(DATA/'trajectories.npz') as cached:
            r={key:(cached[key].item() if cached[key].ndim==0 else cached[key]) for key in cached.files}
    else:
        r=forecast(net)
    figures(net,r)
    print('Figures saved:',OUT,flush=True)
