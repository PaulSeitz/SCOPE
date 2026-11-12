"""RQ2 comparisons with paired absolute returns, using the shared export path."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
from render import COLORS,blend,fmt,family_heading,base_modified_legend
COLORS.update(rows='#7996A8',abspos='#B09970',native_ref='#8D9E77',native_fixed='#BA8F8A')

def generate_rq2(runs,emit):
    conditions=[('seaquest','shift_lanes','Seaquest\nShifted lanes'),('seaquest','random_spawns','Seaquest\nRandom spawns'),('namethisgame','shift_tentacle_spawn_x','Name This Game\nShifted tentacles')]
    allkeys=['ego','abspos','scene','native_ref','rows','native_fixed','oc','pixel']
    limits={e:max((value for k in allkeys for env,c,_ in conditions if env==e for mod in ['base',c] for value in runs.base(e,k,mod).values), default=1)*1.10 for e in ['seaquest','namethisgame']}
    def paired(ax,env,key,condition,pos,vertical=True):
        for mod,offset,tint in [('base',-.18,.65),(condition,.18,0)]:
            sm=runs.base(env,key,mod)
            if not sm.n:continue
            error=np.array([[sm.mean-min(sm.values)],[max(sm.values)-sm.mean]])
            if vertical:
                ax.bar(pos+offset,sm.mean,width=.32,color=blend(COLORS[key],tint),zorder=3)
                ax.errorbar(pos+offset,sm.mean,yerr=error,fmt='none',ecolor='#333333',elinewidth=.45,capsize=1,zorder=4)
            else:
                ax.barh(pos+offset,sm.mean,height=.29,color=blend(COLORS[key],tint),zorder=3)
                ax.errorbar(sm.mean,pos+offset,xerr=error,fmt='none',ecolor='#333333',elinewidth=.45,capsize=1,zorder=4)
    specs=[('fig03_context',['ego','abspos','scene','native_ref'],[0,1,2,3],['SCOPE (A)','+absolute position (E)','+relative objects (C)','+absolute objects (F)'],2.3),('fig03_representation',['scene','rows','native_ref','native_fixed','oc','pixel'],[0,1,2.7,3.7,5.4,6.4],['+relative objects (C)','−dedicated input (D)','+absolute objects (F)','−relative encoding (G)','−rel. encoding, −dedicated (H)','Pixels (I)'],3.0)]
    for stem,methods,xs,labels,height in specs:
        fig,axes=plt.subplots(1,3,figsize=(5.8,height))
        fig.subplots_adjust(left=.10,right=.98,bottom=.40 if height<2.5 else .33,top=.80 if stem=='fig03_representation' else .86,wspace=.30)
        for ax,(env,c,heading) in zip(axes,conditions):
            for x,key in zip(xs,methods):paired(ax,env,key,c,x)
            ax.set_ylim(0,limits[env]);ax.set_xticks(xs,labels,rotation=40,ha='right',rotation_mode='anchor',fontsize=5.8)
            ax.yaxis.set_major_formatter(FuncFormatter(fmt));ax.yaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis='y',labelsize=6)
            ax.grid(axis='y',color='#E9E9E9',lw=.35);ax.set_axisbelow(True)
            ax.set_title(heading,fontsize=7,pad=21 if stem=='fig03_representation' else 6);family_heading(ax,c,1.48 if stem=='fig03_representation' else 1.40)
            if stem=='fig03_representation':
                for center,name in [(0.5,'Dedicated\ninput'),(3.2,'Relative\ntarget'),(5.9,'Neither')]:
                    ax.plot([center-.8,center+.8],[1.015,1.015],transform=ax.get_xaxis_transform(),color='#AAAAAA',lw=.45,clip_on=False)
                    ax.text(center,1.035,name,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=5.4,color='#777777')
        axes[0].set_ylabel('Episode return',fontsize=7)
        base_modified_legend(fig,(.0,1.0),loc='upper left');emit(fig,stem)
    methods=['ego','abs_target','abspos','scene','native_ref','oc','pixel']
    labels=['SCOPE (A)','−relative encoding (J)','+absolute position (E)','+relative objects (C)','+absolute objects (F)','−rel. encoding, −dedicated (H)','Pixels (I)']
    fig,axes=plt.subplots(1,2,figsize=(5.8,2.6));fig.subplots_adjust(left=.26,right=.98,bottom=.20,top=.83,wspace=.35)
    for col,(ax,(env,c,heading)) in enumerate(zip(axes,[('namethisgame','no_shark','NTG · shark removed'),('seaquest','no_enemies','SQ · enemies removed')])):
        for y,key in enumerate(methods):paired(ax,env,key,c,y,False)
        ax.set_yticks(range(len(methods)),labels if col==0 else [],fontsize=6.2);ax.set_ylim(len(methods)-.4,-.6)
        maximum=max(max(runs.base(env,k,m).values) for k in methods for m in ['base',c]);ax.set_xlim(0,maximum*1.12)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt));ax.xaxis.set_major_locator(MaxNLocator(3))
        if c=='no_enemies':ax.set_xscale('symlog',linthresh=100);ax.set_xticks([0,1000,100000,1000000])
        ax.grid(axis='x',color='#E9E9E9',lw=.35);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=6)
        ax.set_title(heading,fontsize=7);family_heading(ax,c,1.30);ax.set_xlabel('Episode return',fontsize=7)
    base_modified_legend(fig,(.0,1.0),loc='upper left')
    emit(fig,'fig03_restriction')


def generate_removals(runs,emit):
    """Main RQ2 figure: SCOPE and four removals, restricted inputs first (story_decisions 2026-11-11)."""
    from render import RQ2_MAIN,RQ2_MAIN_LABELS
    panels=[('seaquest','shift_lanes','Seaquest\nshifted lanes'),('seaquest','random_spawns','Seaquest\nrandom spawns'),
            ('namethisgame','shift_tentacle_spawn_x','Name This Game\nshifted tentacles'),('namethisgame','no_shark','Name This Game\nshark removed')]
    xs=[0,1,2,3.4,4.4]
    labels=[RQ2_MAIN_LABELS[k].replace('restriction, ','restriction,\n') for k in RQ2_MAIN]
    limits={}
    for env,c,_ in panels:
        values=[v for k in RQ2_MAIN for mod in ['base',c] for v in runs.base(env,k,mod).values]
        limits[env]=max(limits.get(env,0),max(values)*1.08)
    fig,axes=plt.subplots(1,4,figsize=(5.8,2.55))
    fig.subplots_adjust(left=.075,right=.995,bottom=.36,top=.74,wspace=.32)
    for i,(ax,(env,c,heading)) in enumerate(zip(axes,panels)):
        for mod,offset,tint in [('base',-.19,.65),(c,.19,0)]:
            for x,key in zip(xs,RQ2_MAIN):
                sm=runs.base(env,key,mod)
                if not sm.n:raise ValueError(f'missing {env} {key} {mod}')
                error=np.array([[sm.mean-min(sm.values)],[max(sm.values)-sm.mean]])
                ax.bar(x+offset,sm.mean,width=.36,color=blend(COLORS[key],tint),zorder=3)
                ax.errorbar(x+offset,sm.mean,yerr=error,fmt='none',ecolor='#333333',elinewidth=.45,capsize=1,zorder=4)
        ax.set_xlim(-.6,5.0);ax.set_ylim(0,limits[env])
        ax.set_xticks(xs,labels,rotation=40,ha='right',rotation_mode='anchor',fontsize=5.6)
        for tick,key in zip(ax.get_xticklabels(),RQ2_MAIN):
            if key=='ego':tick.set_color('#D55E00');tick.set_fontweight('bold')
        ax.yaxis.set_major_formatter(FuncFormatter(fmt));ax.yaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis='y',labelsize=6)
        ax.grid(axis='y',color='#E9E9E9',lw=.35);ax.set_axisbelow(True)
        ax.set_title(heading,fontsize=6.6,pad=17,linespacing=1.1)
        for (a,b),name in [((0,2),'restricted'),((3.4,4.4),'other objects')]:
            ax.plot([a-.32,b+.32],[1.02,1.02],transform=ax.get_xaxis_transform(),color='#AAAAAA',lw=.45,clip_on=False)
            ax.text((a+b)/2,1.035,name,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=5.2,color='#777777')
        ax.axvline(2.7,color='#DDDDDD',lw=.45,zorder=1)
    for ax in (axes[0],axes[2]):ax.set_ylabel('Episode return',fontsize=6.6,labelpad=2)
    base_modified_legend(fig,(.995,1.0))
    emit(fig,'fig03_removals')
