"""Compact drafts for the remaining evaluation figures, using the main export path."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from render import COLORS, blend, fmt, family_heading, base_modified_legend, base_modified_handles
from matplotlib.lines import Line2D
from data import NAMES, TARGET_ORDER

COLORS['tandem_oc_heu']='#548E88'
LABEL={'ego':'SCOPE','pqn_pixel':'PQN pixels','hrl_pixel':'HRL pixels','hrl_oc':'HRL OC','symbolic_pixel':'NEXUS symbolic','tandem_oc':'OC tandem','tandem_pixel':'Pixel tandem','tandem_oc_heu':'Same skills, heuristic'}
def tidy(ax,vertical=False):
    axis=ax.yaxis if vertical else ax.xaxis
    axis.set_major_formatter(FuncFormatter(fmt));axis.set_major_locator(MaxNLocator(3))
    ax.grid(axis='y' if vertical else 'x',color='#E9E9E9',lw=.35);ax.set_axisbelow(True)
    ax.tick_params(labelsize=6,pad=2)
def bar(ax,y,mean,values,color,offset=0):
    ax.barh(y+offset,mean,height=.27 if offset else .48,color=color,zorder=3)
    if values:
        ax.errorbar(mean,y+offset,xerr=[[max(0,mean-min(values))],[max(0,max(values)-mean)]],fmt='none',ecolor='#333333',elinewidth=.45,capsize=1,zorder=4)
def comparison(runs,ax,env,condition,methods,labels=True,base=True):
    maximum=0
    for y,key in enumerate(methods):
        for mod,off,tint in ([('base',-.16,.65),(condition,.16,0)] if base else [(condition,0,0)]):
            s=(runs.transferred('tandem_ocnative_skills',env,env,mod) if key=='tandem_oc_heu' else runs.base(env,key,mod))
            if not s.n:raise ValueError(f'Missing plotted result: {env}/{key}/{mod}')
            bar(ax,y,s.mean,s.values,blend(COLORS.get(key,'#777777'),tint),off)
            maximum=max(maximum,max(s.values))
    ax.set_yticks(range(len(methods)),[LABEL[k] for k in methods] if labels else [],fontsize=6.2)
    ax.set_ylim(len(methods)-.6,-.6);ax.set_xlim(0,max(1,maximum)*1.12);tidy(ax)
    if condition=='no_enemies':
        ax.set_xscale('symlog',linthresh=100);ax.set_xticks([0,1000,100000,1000000])
    for tick,key in zip(ax.get_yticklabels(),methods):
        if key=='ego':tick.set_color(COLORS['ego']);tick.set_weight('bold')
def metric(runs,key,condition):
    if (key,condition) in runs.rescue_rerun:  # NEXUS/HRL: counters from the separate 2026-11-12 rerun (data.RESCUE_RERUN)
        return [o.mean for o in runs.rescue_rerun[key,condition]]
    means=[]
    for obs in runs.home['seaquest',key,condition]:
        raw=json.loads(Path(obs.path).read_text())['results'][condition]
        eps=raw.get('episode_metrics',[])
        if eps and all('env_rescues' in ep for ep in eps):means.append(float(np.mean([ep['env_rescues'] for ep in eps])))
    return means

RESCUE_INK='#1F1F1F'
def rescue_axis(runs,ax,methods,condition):
    """Secondary x-axis on a horizontal return panel (supervisor feedback #9): successful rescues per episode as
    diamonds, hollow for the base game and filled under the modification, on the same rows as the return bars."""
    top=ax.twiny();largest=0
    for y,key in enumerate(methods):
        for mod,off,fill in [('base',-.16,'white'),(condition,.16,RESCUE_INK)]:
            vs=metric(runs,key,mod)
            if not vs:continue
            m=float(np.mean(vs));largest=max(largest,max(vs))
            top.plot(m,y+off,marker='D',ms=3.2,mfc=fill,mec=RESCUE_INK,mew=.6,ls='none',zorder=7)
    top.set_xlim(0,largest*1.12);top.xaxis.set_major_locator(MaxNLocator(3));top.tick_params(axis='x',labelsize=6,pad=1,colors='#333333')
    top.spines['top'].set_visible(True);top.spines['top'].set_color('#888888');top.spines['top'].set_linewidth(.45)
    top.set_xlabel('Successful rescues per episode',fontsize=6.2,labelpad=3,color='#333333')
    return top

def rescue_handles():
    return [Line2D([],[],marker='D',ms=3.2,mfc='white',mec=RESCUE_INK,mew=.6,ls='none',label='rescues, base'),
            Line2D([],[],marker='D',ms=3.2,mfc=RESCUE_INK,mec=RESCUE_INK,mew=.6,ls='none',label='rescues, modified')]

def generate_remaining(runs,emit):
    # Two complete rows; slower-player NTG fills the fourth condition slot.
    fig,axes=plt.subplots(2,4,figsize=(5.8,3.35))
    fig.subplots_adjust(left=.12,right=.99,bottom=.17,top=.86,wspace=.28,hspace=1.45)
    configs=[('seaquest',[('shift_lanes','Shifted lanes'),('random_spawns','Random spawns'),('faster_enemies','Faster enemies'),('change_background_color','Background')]),('namethisgame',[('shift_tentacle_spawn_x','Shifted tentacles'),('faster_shark','Faster shark'),('slower_player','Slower player'),('change_background_color','Background')])]
    for row,(env,conditions) in enumerate(configs):
        methods=['ego','pqn_pixel','hrl_pixel','hrl_oc']+(['symbolic_pixel'] if env=='seaquest' else [])
        maximum=max(max(runs.base(env,key,condition).values) for key in methods for condition in ['base']+[c for c,_ in conditions])
        for col,(condition,title) in enumerate(conditions):
            ax=axes[row,col]
            for x,key in enumerate(methods):
                for mod,offset,tint in [('base',-.18,.65),(condition,.18,0)]:
                    sm=runs.base(env,key,mod)
                    ax.bar(x+offset,sm.mean,width=.32,color=blend(COLORS[key],tint),zorder=3)
                    ax.errorbar(x+offset,sm.mean,yerr=[[sm.mean-min(sm.values)],[max(sm.values)-sm.mean]],fmt='none',ecolor='#333333',elinewidth=.45,capsize=1,zorder=4)
            ax.set_ylim(0,maximum*1.12);tidy(ax,vertical=True)
            ax.set_xticks(range(len(methods)),[{'ego':'SCOPE','pqn_pixel':'PQN pix.','hrl_pixel':'HRL pix.','hrl_oc':'HRL OC','symbolic_pixel':'NEXUS'}.get(k,k) for k in methods],rotation=45,ha='right',fontsize=5.8)
            ax.set_title(title,fontsize=6.4,pad=6);family_heading(ax,condition,1.34)
            if col:ax.tick_params(axis='y',labelleft=False)
            else:ax.set_ylabel('Episode return',fontsize=6.3)
            for tick,key in zip(ax.get_xticklabels(),methods):
                if key=='ego':tick.set_color(COLORS[key]);tick.set_weight('bold')
        bounds=axes[row,0].get_position()
        fig.text(.012,(bounds.y0+bounds.y1)/2,NAMES[env],rotation=90,ha='center',va='center',fontsize=7.3,weight='semibold')
    base_modified_legend(fig,(.995,.995))
    emit(fig,'fig02_system_compact')
    # Task simplification: return bars on the bottom axis, Seaquest rescues on a secondary top axis.
    fig,axes=plt.subplots(1,2,figsize=(5.8,2.45));fig.subplots_adjust(left=.17,right=.98,top=.66,bottom=.17,wspace=.12)
    for ax,env,condition,methods,title in [(axes[0],'seaquest','no_enemies',['ego','pqn_pixel','hrl_pixel','hrl_oc','symbolic_pixel'],'Seaquest, no enemies'),(axes[1],'namethisgame','no_shark',['ego','pqn_pixel','hrl_pixel','hrl_oc'],'Name This Game, no shark')]:
        comparison(runs,ax,env,condition,methods,labels=ax==axes[0]);ax.set_ylim(4.5,-.6);ax.set_xlabel('Episode return',fontsize=6.2)
        if env=='seaquest':
            rescue_axis(runs,ax,methods,condition)
            ax.set_title(title,fontsize=7,pad=28)
        else:
            ax.set_title(title,fontsize=7,pad=28)
    fig.legend(handles=base_modified_handles()+rescue_handles(),loc='upper center',bbox_to_anchor=(.575,1.0),ncol=4,fontsize=6,frameon=False,handlelength=1.0,handleheight=.8,handletextpad=.35,columnspacing=1.1,borderaxespad=0)
    emit(fig,'fig02_task_rescues')
    # Transfer: every directed source-target pair, divided into two compact figures.
    for group,targets in enumerate([TARGET_ORDER[:4],TARGET_ORDER[4:]],1):
        fig=plt.figure(figsize=(5.8,3.45));cols=2 if len(targets)==4 else 3;w=.34 if cols==2 else .20;gap=.16 if cols==2 else .12
        for j,target in enumerate(targets):
            row,col=divmod(j,cols);shift=.16 if cols==3 and row==1 else 0;ax=fig.add_axes([.14+shift+col*(w+gap),.60 if row==0 else .13,w,.27])
            entries=[]
            for source in ['seaquest','namethisgame']:
                sm=runs.transferred('skills_ego',source,target) if source!=target else None
                entries.append((sm.mean if sm and sm.n else None,sm.values if sm and sm.n else [],COLORS['ego' if source=='seaquest' else 'screen']))
            sm=runs.base(target,'ego');entries.append((sm.mean,sm.values,'#8C9399'))
            for kind,color in [('random_actions','#DADDE0'),('heuristic_random','#AEB5BA'),('fixed_ego_rules','#596A76')]:
                c=runs.control(target,kind);entries.append((c['mean'] if c else None,[],color))
            for y,(mean,vs,color) in enumerate(entries):
                if mean is not None:bar(ax,y,mean,vs,color)
                else:ax.text(.02,y,'n/a',transform=ax.get_yaxis_transform(),va='center',fontsize=5.5,color='#888888')
            ax.set_yticks(range(6),['SQ skills','NTG skills','Native','Random actions','Untrained skills','Fixed rules'] if col==0 else [],fontsize=5.8);ax.set_ylim(5.6,-.6);tidy(ax);ax.set_title(NAMES[target],fontsize=7)
        fig.supxlabel('Episode return',fontsize=7,y=.005)
        emit(fig,f'fig04_transfer_{group}')
    # Tandem comparison focused on distinct failures, preserving the usual bars.
    fig=plt.figure(figsize=(5.8,3.2));conditions=[('seaquest','base','Base'),('seaquest','shift_lanes','Shifted lanes'),('seaquest','random_spawns','Random spawns'),('seaquest','change_background_color','Background'),('namethisgame','base','Base'),('namethisgame','shift_tentacle_spawn_x','Shifted tentacles'),('namethisgame','no_shark','No shark'),('namethisgame','change_background_color','Background')]
    for j,(env,condition,title) in enumerate(conditions):
        row,col=divmod(j,4);ax=fig.add_axes([.22+col*.20,.61 if row==0 else .14,.175,.27])
        comparison(runs,ax,env,condition,['ego','tandem_oc','tandem_pixel'],labels=col==0,base=condition!='base');ax.set_title(title,fontsize=6.4,pad=7);family_heading(ax,condition,1.33)
    fig.text(.015,.745,'Seaquest',rotation=90,ha='center',va='center',fontsize=7.5,weight='semibold');fig.text(.015,.275,'Name This Game',rotation=90,ha='center',va='center',fontsize=7.5,weight='semibold')
    fig.supxlabel('Episode return',fontsize=7,y=.015)
    base_modified_legend(fig,(.995,.012),loc='lower right')
    emit(fig,'fig05_tandem_compact')
    fig,axes=plt.subplots(1,3,figsize=(5.8,2.25));fig.subplots_adjust(left=.25,right=.98,top=.64,bottom=.18,wspace=.35)
    for j,(ax,condition,title) in enumerate(zip(axes,['base','random_spawns','no_enemies'],['a) Base','b) Random spawns','c) No enemies'])):
        comparison(runs,ax,'seaquest',condition,['ego','tandem_oc','tandem_oc_heu'],labels=j==0,base=False);ax.set_title(title,fontsize=7,pad=27);ax.set_xlabel('Episode return',fontsize=6.2);family_heading(ax,condition,1.62)
        if condition=='no_enemies':
            ax.set_xscale('linear');ax.set_xlim(0,1.10*max(max(runs.base('seaquest',key,condition).values) for key in ['ego','tandem_oc']));ax.xaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_formatter(FuncFormatter(fmt))
            rescue_axis(runs,ax,['ego','tandem_oc','tandem_oc_heu'],condition)
    emit(fig,'fig05_selector_recovery')
