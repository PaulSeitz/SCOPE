"""Compact method illustrations from the previously saved example frames/state."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent

def generate_examples(emit):
    fig,axes=plt.subplots(1,2,figsize=(4.0,1.75))
    for ax,game,name in zip(axes,['seaquest','namethisgame'],['Seaquest','Name This Game']):
        image=np.load(HERE/'environment_frames'/f'{game}.npy')
        ax.imshow(image[30:190,:160],interpolation='nearest');ax.axis('off');ax.set_title(name,fontsize=8,pad=3)
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.84,wspace=.12)
    emit(fig,'fig_environments')
    s=np.load(HERE/'assets/worked_example_state.npz');target=int(s['target']);F=s['frame']
    px,py,pw,ph=[float(s[k]) for k in ['px','py','pw','ph']]
    tx,ty,tw,th=[float(s[k][target]) for k in ['x','y','w','h']]
    cx,cy=px+pw/2,py+ph/2;tcx,tcy=tx+tw/2,ty+th/2
    sign=-1 if round(float(s['f']))==270 else 1
    fig=plt.figure(figsize=(5.8,1.85));gs=fig.add_gridspec(1,4,left=.02,right=.99,bottom=.24,top=.82,wspace=.25)
    ax=fig.add_subplot(gs[0]);ax.imshow(F,interpolation='nearest');ax.set_xlim(0,160);ax.set_ylim(190,30);ax.set_aspect('equal');ax.axis('off')
    ax.add_patch(Rectangle((tx-2,ty-2),tw+4,th+4,fill=False,ec='white',lw=1))
    ax.set_title('(a) Full-scene selection',fontsize=6.4,pad=8)
    polar=fig.add_subplot(gs[1],projection='polar');bins=s['bins'];theta=np.arange(8)*np.pi/4
    polar.set_theta_zero_location('N');polar.set_theta_direction(-1);polar.bar(theta,bins,width=np.pi/4*.8,color='#D55E00')
    polar.set_ylim(0,1);polar.set_yticks([]);polar.set_xticks(theta);polar.set_xticklabels(['forward','','','','behind','','',''],fontsize=5.5);polar.tick_params(pad=-2);polar.grid(color='#DDDDDD',lw=.4)
    polar.set_title('(b) Restricted input',fontsize=6.4,pad=12)
    mx,my=(cx+tcx)/2,(cy+tcy)/2;x0=int(np.clip(mx-36,0,88));y0=int(np.clip(my-36,0,138))
    for j in range(2):
        ax=fig.add_subplot(gs[j+2]);mir=j==1;image=F[:,::-1] if mir else F;left=160-x0-72 if mir else x0
        ax.imshow(image[y0:y0+72,left:left+72],interpolation='nearest');ax.axis('off');ax.set_aspect('equal')
        targetx=160-tx-tw-left if mir else tx-left;playerx=160-cx-left if mir else cx-left
        ax.add_patch(Rectangle((targetx-2,ty-y0-2),tw+4,th+4,fill=False,ec='white',lw=1))
        facing=-sign if mir else sign;direction='RIGHT' if facing==1 else 'LEFT'
        ax.annotate('',xy=(playerx+facing*14,cy-y0),xytext=(playerx,cy-y0),arrowprops=dict(arrowstyle='-|>',color='#56D9FF',lw=1,mutation_scale=8))
        ax.set_title('(c) Action translation' if j==0 else 'Mirrored arrangement',fontsize=6.4,pad=8)
        ax.text(.5,-.10,f'FORWARD → {direction}',transform=ax.transAxes,ha='center',va='top',fontsize=6)
    # The actual input values are labels, not narrative explanations.
    bounds=polar.get_position();fig.text((bounds.x0+bounds.x1)/2,.12,f'd = {float(s["d"]):.2f}',ha='center',fontsize=6)
    emit(fig,'fig_worked_example')
