"""Rendering only: all numerical values come from data.Runs."""
from __future__ import annotations
import math
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
from data import HOMES, NAMES, TARGET_ORDER, CONTROL_KINDS

COLORS={'ego':'#D55E00','screen':'#548E88','oc':'#6C819D','pixel':'#9B7BA1','pqn_pixel':'#6E8295','pqn_oc':'#969FA7','hrl_pixel':'#8D8575','hrl_oc':'#A69A89','neural_pixel':'#8C9D82','neural_oc':'#A4AE9A','nesy_pixel':'#5C8279','nesy_oc':'#8DA69C','symbolic_pixel':'#90915B','symbolic_oc':'#AFAD86','tandem_oc':'#009E73','tandem_pixel':'#8064B3','pqn_oc_large':'#7E909C','nocross':'#8F9E89'}
LABELS={'ego':'Reference ego','screen':'$-$facing','oc':'+absolute objects, $-$relative encoding, $-$dedicated target input','pixel':'Pixels','pqn_pixel':'PQN · pixels','pqn_oc':'PQN · native OC','hrl_pixel':'HRL · pixels','hrl_oc':'HRL · OC','neural_pixel':'NEXUS neural · pixels','neural_oc':'NEXUS neural · OC','nesy_pixel':'NEXUS neurosymbolic · pixels','nesy_oc':'NEXUS neurosymbolic · OC','symbolic_pixel':'NEXUS symbolic · pixels','symbolic_oc':'NEXUS symbolic · OC','tandem_oc':'OC tandem','tandem_pixel':'Pixel tandem','pqn_oc_large':'OC PQN 3×128','nocross':'Acting-skill-only'}
SHORT={'ego':'Ref','screen':'Screen','oc':'OC skills','pixel':'Pixel skills','pqn_pixel':'PQN P','pqn_oc':'PQN O','hrl_pixel':'HRL P','hrl_oc':'HRL O','neural_pixel':'N neural','nesy_pixel':'N neSy','symbolic_pixel':'N sym','tandem_oc':'OC tandem','tandem_pixel':'Pixel tandem'}
SYSTEM={'seaquest':['ego','pqn_pixel','hrl_pixel','symbolic_pixel'],'namethisgame':['ego','pqn_pixel','hrl_pixel']}
SUPPLEMENTARY_OC={e:['ego','pqn_oc','hrl_oc'] for e in HOMES}
FULL_SYSTEM={'seaquest':['ego','pqn_pixel','pqn_oc','hrl_pixel','hrl_oc','neural_pixel','neural_oc','nesy_pixel','nesy_oc','symbolic_pixel','symbolic_oc'],'namethisgame':['ego','pqn_pixel','pqn_oc','hrl_pixel','hrl_oc']}
# Base performance: SCOPE and its four main ablations in the order of fig03_skills, then the main baselines in the order of fig02_system after a divider ('|').
# Main-text RQ2 inputs, fixed order and labels (author, 2026-11-11). Appendix assets keep letters.
RQ2_MAIN=['ego','screen','abs_target','native_ref','native_fixed']
RQ2_MAIN_LABELS={'ego':'SCOPE','screen':'\u2212facing','abs_target':'\u2212relative encoding','native_ref':'\u2212restriction','native_fixed':'\u2212restriction, \u2212relative encoding'}
COMPETENCE={e:RQ2_MAIN+['|']+[k for k in SYSTEM[e] if k!='ego'] for e in SYSTEM}
SUPPLEMENTARY_COMPETENCE={'seaquest':['ego','pqn_oc','hrl_oc','neural_pixel','neural_oc','nesy_pixel','nesy_oc','symbolic_oc'],'namethisgame':['ego','pqn_oc','hrl_oc']}
SHORT.update({'ego':'Ref','neural_oc':'Neural O','nesy_oc':'NeSy O','symbolic_oc':'Sym O'})
COLORS.update({'scene':'#C8A02B', 'abs_target':'#5B8DB8'})
# Shared with render_rq2 so that every figure using these arms has a color, independent of import order.
COLORS.update(rows='#7996A8',abspos='#B09970',native_ref='#8D9E77',native_fixed='#BA8F8A')
LABELS.update({'scene':'+relative objects','ego':'SCOPE', 'abs_target':'$-$relative encoding'})
SHORT.update({'scene':'Scene','ego':'SCOPE', 'abs_target':'−rel. enc.'})
SKILLS={e:['ego','screen','scene','abs_target','oc','pixel'] for e in HOMES}
# Ablations are named by what changes relative to SCOPE (approved by the author, 2026-11-06).
ABLATION_COLOR='#7A7F85'
ABLATION_LABELS={'ego':'SCOPE','screen':'\u2212facing','scene':'+relative objects','oc':'+absolute objects,\n\u2212rel. encoding, \u2212dedicated','pixel':'pixels'}
ABLATION_LABELS['abs_target']='\u2212relative encoding'
ABLATION_KEYS={'abs_target','screen','scene','oc','pixel'}
MOD_NAMES={'no_shark':'No shark','random_spawns':'Random spawns','shift_lanes':'Shifted spawn lanes','faster_enemies':'Faster enemies','no_enemies':'No enemies','change_background_color':'Changed background','shift_tentacle_spawn_x':'Shifted tentacle spawns','faster_shark':'Faster shark','no_shark_short_tentacles':'No shark, short tentacles'}
# Physical width matches the thesis text block; fonts no longer shrink to 3 pt.
PAPER_WIDTH = 5.8
LEGEND_LABELS = dict(LABELS, ego='SCOPE', scene='+relative objects', pqn_pixel='PQN pixels', pqn_oc='PQN OC', hrl_pixel='HRL pixels', hrl_oc='HRL OC', neural_pixel='NEXUS neural', nesy_pixel='NEXUS neSy', symbolic_pixel='NEXUS symbolic', neural_oc='NEXUS neural OC', nesy_oc='NEXUS neSy OC', symbolic_oc='NEXUS symbolic OC')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aaaaaa','axes.labelcolor':'#444444','xtick.color':'#555555','ytick.color':'#555555','axes.linewidth':.45,'xtick.major.width':.45,'ytick.major.width':.45,'xtick.major.size':2,'ytick.major.size':2,'axes.titleweight':'medium','svg.fonttype':'none','svg.hashsalt':'thesis-evaluation-v1','pdf.fonttype':42})


def modification_family(condition):
    if condition in ('shift_lanes','random_spawns','shift_tentacle_spawn_x'):return 'Layout'
    if condition in ('faster_enemies','gravity','faster_shark','slower_player'):return 'Dynamics'
    if condition in ('no_enemies','no_shark','no_shark_short_tentacles'):return 'Task structure'
    if condition=='change_background_color':return 'Appearance'
    return ''

def family_heading(ax, condition, y=1.34):
    family=modification_family(condition)
    if family:ax.text(.5,1+(y-1)*.70,family,transform=ax.transAxes,ha='center',va='bottom',fontsize=6,color='#888888')

BASE_KEY_COLOR='#7A7F85'
def base_modified_handles(color=BASE_KEY_COLOR):
    """Light/dark key for paired bars (supervisor feedback #11): light = base game, saturated = modified game."""
    return [Patch(facecolor=blend(color,.65),label='base'),Patch(facecolor=color,label='modified')]

def base_modified_legend(fig,anchor,loc='upper right',fontsize=6):
    return fig.legend(handles=base_modified_handles(),loc=loc,bbox_to_anchor=anchor,ncol=2,fontsize=fontsize,frameon=False,
                      handlelength=1.0,handleheight=.8,handletextpad=.35,columnspacing=.9,borderaxespad=0)

def fmt(x,pos=None):
    if abs(x)>=1e6:return f'{x/1e6:g}m'
    if abs(x)>=1000:return f'{x/1000:g}k'
    return f'{x:g}'

def tidy(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(fmt));ax.grid(axis='x',color='#e9e9e9',lw=.35);ax.set_axisbelow(True);ax.tick_params(labelsize=6.5,pad=2)

def conditions(env,layout):
    return [layout,'faster_enemies','no_enemies','change_background_color'] if env=='seaquest' else ['shift_tentacle_spawn_x','faster_shark','no_shark','change_background_color']

def blend(c,white):return tuple(np.array(to_rgb(c))*(1-white)+white)
def sd(s):return s.sd or 0.

class Renderer:
    def __init__(self,runs,emit):self.runs=runs;self.emit=emit;self.limits=self._limits()
    def _limits(self):
        keys=set(SYSTEM['seaquest']+SKILLS['seaquest']+['tandem_oc','tandem_pixel'])
        result={}
        for env in HOMES:
            upper=[]
            for k in set(FULL_SYSTEM[env]+SKILLS[env]+['tandem_oc','tandem_pixel']):
                for m in (['base','random_spawns','shift_lanes','faster_enemies','change_background_color'] if env=='seaquest' else ['base','shift_tentacle_spawn_x','faster_shark','change_background_color']):
                    s=self.runs.base(env,k,m)
                    if s.n:upper.append(s.mean+sd(s))
            result[env]=math.ceil(max(upper,default=10000)*1.10/10000)*10000
        return result

    @staticmethod
    def _row_label(fig, axes, env, x=.018):
        bounds=axes.get_position()
        fig.text(x,(bounds.y0+bounds.y1)/2,NAMES[env],rotation=90,
                 ha='center',va='center',fontsize=8,fontweight='semibold',color='#333333')

    @staticmethod
    def _reference_ticks(ax,order):
        for tick,key in zip(ax.get_xticklabels(),order):
            if key=='ego':tick.set_fontweight('bold');tick.set_color(COLORS['ego'])

    @staticmethod
    def _reference_legend(legend,keys):
        for label,key in zip(legend.get_texts(),keys):
            if key=='ego':label.set_fontweight('bold');label.set_color(COLORS['ego'])

    def competence(self,orders=None,stem='fig01_competence'):
        """Base returns per game. A '|' in an order separates two groups with a thin divider."""
        orders=COMPETENCE if orders is None else orders
        pitch,bar,gap=.2,.46,.42  # inches per row, bar height in rows, inches between the two games
        rows={e:[k for k in orders[e] if k!='|'] for e in HOMES}
        heights=[len(rows[e]) for e in HOMES]
        bottom,top=.42,.06
        height=bottom+top+gap+pitch*sum(heights)
        fig,axes=plt.subplots(2,1,figsize=(PAPER_WIDTH,height),gridspec_kw={'height_ratios':heights,'hspace':gap/(pitch*sum(heights)/2)})
        fig.subplots_adjust(left=.40,right=.97,top=1-top/height,bottom=bottom/height)
        for ax,env in zip(axes,HOMES):
            order=rows[env]
            for y,k in enumerate(order):
                summary=self.runs.base(env,k)
                if not summary.n:
                    ax.text(.02,y,'TBD',transform=ax.get_yaxis_transform(),va='center',color='#888888',fontstyle='italic',fontsize=6.5)
                    continue
                self._base_bar(ax,y,summary,COLORS[k],bar)
            if '|' in orders[env]:
                ax.axhline(orders[env].index('|')-.5,color='#bbbbbb',lw=.45,ls=(0,(2,2)),zorder=1)
            def label(k):
                text=RQ2_MAIN_LABELS[k] if k in RQ2_MAIN_LABELS else ABLATION_LABELS[k] if k in ABLATION_KEYS else LABELS[k].replace('NEXUS neurosymbolic','NEXUS neSy')
                n=self.runs.base(env,k).n
                return text+(f'  n={n}' if n not in [0,5] else '')
            ax.set_yticks(range(len(order)),[label(k) for k in order],fontsize=7)
            ax.set_ylim(len(order)-.5,-.5);ax.set_xlim(left=0);tidy(ax)
            ax.set_xlabel('Episode return',fontsize=7,labelpad=3)
            for tick,k in zip(ax.get_yticklabels(),order):
                if k=='ego':tick.set_color(COLORS[k]);tick.set_fontweight('bold')
            self._row_label(fig,ax,env,x=.012)
        self.emit(fig,stem)

    @staticmethod
    def _base_bar(ax,y,summary,color,height):
        """One horizontal base bar. Whiskers span the lowest and highest checkpoint, as in the modification panels."""
        ax.barh(y,summary.mean,height=height,color=color,edgecolor='none',zorder=3)
        if len(summary.values)>1:
            ax.errorbar(summary.mean,y,xerr=np.array([[summary.mean-min(summary.values)],[max(summary.values)-summary.mean]]),fmt='none',ecolor='#222222',elinewidth=.5,capsize=1.4,capthick=.5,zorder=4)

    def _bars(self,ax,env,mod,order,first=False,narrow=False):
        if mod is None:
            ax.axis('off');ax.text(.5,.5,'No appearance-only test\nin this snapshot',ha='center',va='center',color='#888888',transform=ax.transAxes,fontsize=6.5);return
        ymax=0
        for i,k in enumerate(order):
            for off,m,light in [(-.19,'base',True),(.19,mod,False)]:
                s=self.runs.base(env,k,m)
                if not s.n:
                    if not light:ax.text(i,.02,'TBD',transform=ax.get_xaxis_transform(),rotation=90,ha='center',fontsize=6,color='#999999')
                    continue
                ymax=max(ymax,*s.values);color=blend(COLORS[k],.68 if len(order)>5 else .64) if light else COLORS[k]
                ax.bar(i+off,s.mean,width=.34,color=color,edgecolor='none',zorder=3)
                if len(s.values)>1:ax.errorbar(i+off,s.mean,yerr=np.array([[s.mean-min(s.values)],[max(s.values)-s.mean]]),fmt='none',ecolor='#222222',elinewidth=.5,capsize=1.4,capthick=.5,zorder=4,snap=True)
        if mod=='no_enemies':
            ax.set_yscale('symlog',linthresh=100,linscale=.65);ax.set_ylim(0,max(2e6,ymax*1.1));ax.set_yticks([0,1000,10000,100000,1000000]);suffix=''
        else:
            ax.set_ylim(0,max(self.limits[env],math.ceil(ymax*1.08/10000)*10000,10000));suffix=''
        axis_labels=[{'pqn_pixel':'PQN','hrl_pixel':'HRL','symbolic_pixel':'NEXUS'}.get(k,SHORT[k]) for k in order] if set(order).issubset({'ego','pqn_pixel','hrl_pixel','symbolic_pixel'}) else [SHORT[k] for k in order]
        ax.set_xticks(range(len(order)),axis_labels,rotation=60,ha='right',fontsize=7);self._reference_ticks(ax,order);ax.yaxis.set_major_formatter(FuncFormatter(fmt));ax.tick_params(axis='y',labelsize=6.7,pad=2);ax.grid(axis='y',color='#e9e9e9',lw=.35);ax.set_axisbelow(True)
        ax._eval_heading = {'shift_lanes':'Shifted spawn\nlanes','change_background_color':'Changed\nbackground','shift_tentacle_spawn_x':'Shifted tentacle\nspawns','no_shark_short_tentacles':'No shark /\nshort tentacles','no_shark':'No shark'}.get(mod,MOD_NAMES[mod])
        ax._eval_log_note = mod=='no_enemies'

    @staticmethod
    def _place_headings(fig,axes,categories=False):
        # Use one shared physical header band: first lines have identical anchors
        # regardless of wrapping, and both heading tiers share each axis centre.
        height_pt=fig.get_figheight()*72
        for row in axes:
            for ax in row:
                title=getattr(ax,'_eval_heading','')
                if not title:continue
                bounds=ax.get_position(); centre=(bounds.x0+bounds.x1)/2
                header=fig.text(centre,bounds.y1+26/height_pt,title,
                    ha='center',va='top',fontsize=7.4,linespacing=1.16)
                ax._eval_header_texts=[header]
                if getattr(ax,'_eval_log_note',False):
                    note=fig.text(centre,bounds.y1+14/height_pt,'(log scale)',
                        ha='center',va='top',fontsize=6.2,fontstyle='italic',color='#666666')
                    ax._eval_header_texts.append(note)
        if categories:
            for ax,label in zip(axes[0,-4:],['Layout','Dynamics','Task structure','Appearance']):
                bounds=ax.get_position()
                fig.text((bounds.x0+bounds.x1)/2,bounds.y1+43/height_pt,label,
                         ha='center',va='top',fontsize=7.1,color='#777777')

    @staticmethod
    def _check_row_spacing(fig,axes):
        # Check drawn text, not just axis rectangles: long rotated tick labels
        # must clear the next row's multiline titles at the final font size.
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        ticks=[label.get_window_extent(renderer).y0 for ax in axes[0]
               if ax.axison for label in ax.get_xticklabels() if label.get_visible()]
        titles=[text.get_window_extent(renderer).y1 for ax in axes[1]
                for text in getattr(ax,'_eval_header_texts',[])]
        clearance=(min(ticks)-max(titles))*72/fig.dpi
        if clearance<3:
            raise ValueError(f'Row labels collide or have insufficient clearance: {clearance:.2f} pt')

    @staticmethod
    def _check_legend_spacing(fig,axes,legend):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        ticks=[label.get_window_extent(renderer).y0 for ax in axes[-1]
               if ax.axison for label in ax.get_xticklabels() if label.get_visible()]
        clearance=(min(ticks)-legend.get_window_extent(renderer).y1)*72/fig.dpi
        if clearance<3:
            raise ValueError(f'Legend overlaps bottom-row labels: {clearance:.2f} pt')

    def robustness(self,orders,stem,layout,rows=None):
        rows=[(env,orders[env],NAMES[env]) for env in HOMES] if rows is None else rows
        keys=list(dict.fromkeys(key for env,order,label in rows for key in order))
        bottom=.23 if len(keys)>4 else (.215 if any(key in keys for key in ['screen','oc','pixel']) else .17)
        bottom+=.045 if len(keys)+2>8 or len(keys)<=4 else .0  # room for the base/modified key in the legend
        fig,axes=plt.subplots(2,4,figsize=(PAPER_WIDTH,3.95),gridspec_kw={'wspace':.22,'hspace':1.2})
        fig.subplots_adjust(left=.105,right=.995,top=.90,bottom=bottom)
        for row,(env,order,label) in enumerate(rows):
            for col,mod in enumerate(conditions(env,layout)):
                self._bars(axes[row,col],env,mod,order)
            axes[row,0].set_ylabel('Episode return',fontsize=6.5,labelpad=2)
            bounds=axes[row,0].get_position()
            fig.text(.012,(bounds.y0+bounds.y1)/2,label,rotation=90,ha='center',va='center',fontsize=8,fontweight='semibold',color='#333333')
        self._place_headings(fig,axes,categories=True)
        handles=[Patch(facecolor=COLORS[key],label=LEGEND_LABELS[key]) for key in keys]
        handles+=base_modified_handles()
        legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,-.015),ncol=4,fontsize=6.5,frameon=False,handlelength=1.2,handletextpad=.4,columnspacing=1.1,labelspacing=.55)
        self._reference_legend(legend,keys)
        self._check_row_spacing(fig,axes)
        self._check_legend_spacing(fig,axes,legend)
        self.emit(fig,stem)

    def ablations(self,stem,layout):
        """SCOPE against its ablations. Horizontal paired bars, with the labels once at the left of each game."""
        order=SKILLS['seaquest']
        fig,axes=plt.subplots(2,4,figsize=(PAPER_WIDTH,3.95),gridspec_kw={'wspace':.16,'hspace':.72})
        fig.subplots_adjust(left=.285,right=.985,top=.885,bottom=.075)
        for row,env in enumerate(HOMES):
            for col,mod in enumerate(conditions(env,layout)):
                ax=axes[row,col]
                largest=0
                for y,k in enumerate(order):
                    for off,m,light in [(-.19,'base',True),(.19,mod,False)]:
                        s=self.runs.base(env,k,m)
                        if not s.n:continue
                        largest=max(largest,*s.values)
                        hue=COLORS['ego'] if k=='ego' else ABLATION_COLOR
                        color=blend(hue,.64) if light else hue
                        ax.barh(y+off,s.mean,height=.34,color=color,edgecolor='none',zorder=3)
                        if len(s.values)>1:ax.errorbar(s.mean,y+off,xerr=np.array([[s.mean-min(s.values)],[max(s.values)-s.mean]]),fmt='none',ecolor='#222222',elinewidth=.5,capsize=1.1,capthick=.5,zorder=4)
                if mod=='no_enemies':
                    ax.set_xscale('symlog',linthresh=100,linscale=.65);ax.set_xlim(0,max(2e6,largest*1.1));ax.set_xticks([0,1000,100000])
                else:
                    ax.set_xlim(0,max(self.limits[env],math.ceil(largest*1.08/10000)*10000,10000));ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_ylim(len(order)-.5,-.5)
                ax.set_yticks(range(len(order)),[ABLATION_LABELS[k] for k in order] if col==0 else [],fontsize=6.9,linespacing=1.05)
                ax.xaxis.set_major_formatter(FuncFormatter(fmt));ax.tick_params(axis='x',labelsize=6.5,pad=2);ax.tick_params(axis='y',pad=2,length=2 if col==0 else 0)
                ax.grid(axis='x',color='#e9e9e9',lw=.35);ax.set_axisbelow(True)
                ax._eval_heading={'shift_lanes':'Shifted spawn\nlanes','change_background_color':'Changed\nbackground','shift_tentacle_spawn_x':'Shifted tentacle\nspawns','no_shark':'No shark'}.get(mod,MOD_NAMES[mod])
                ax._eval_log_note=mod=='no_enemies'
                if col==0:
                    tick=ax.get_yticklabels()[0];tick.set_fontweight('bold');tick.set_color(COLORS['ego'])
            self._row_label(fig,axes[row,0],env,x=.012)
        self._place_headings(fig,axes,categories=True)
        for ax in axes[-1]:ax.set_xlabel('Episode return',fontsize=6.5,labelpad=2)
        base_modified_legend(fig,(.012,.995),loc='upper left',fontsize=6.3)
        self._check_row_spacing(fig,axes)
        self.emit(fig,stem)

    def tandem(self,stem,layout):
        order=['ego','tandem_oc','tandem_pixel']
        # Base performance stays at the left of each source row.
        fig=plt.figure(figsize=(PAPER_WIDTH,3.8))
        grid=fig.add_gridspec(2,5,left=.105,right=.995,top=.90,bottom=.17,wspace=.26,hspace=1.05)
        axes=np.array([[fig.add_subplot(grid[r,c]) for c in range(5)] for r in range(2)])
        names=['SCOPE','OC','Pixel']
        for row,env in enumerate(HOMES):
            for col,mod in enumerate(conditions(env,layout)):
                self._bars(axes[row,col+1],env,mod,order,narrow=True)
                axes[row,col+1].set_xticklabels(names if mod else [],rotation=60,ha='right',fontsize=7)
                if mod:self._reference_ticks(axes[row,col+1],order)
            # Base performance stays at the left of each row, on the scale of the linear panels beside it.
            ax=axes[row,0]
            for x,k in enumerate(order):
                summary=self.runs.base(env,k)
                if not summary.n:
                    ax.text(x,.02,'TBD',transform=ax.get_xaxis_transform(),rotation=90,ha='center',fontsize=6,color='#999999')
                    continue
                ax.bar(x,summary.mean,width=.58,color=blend(COLORS[k],.64),edgecolor='none',zorder=3)
                if len(summary.values)>1:ax.errorbar(x,summary.mean,yerr=np.array([[summary.mean-min(summary.values)],[max(summary.values)-summary.mean]]),fmt='none',ecolor='#222222',elinewidth=.5,capsize=1.4,capthick=.5,zorder=4)
            ax.set_ylim(axes[row,1].get_ylim())
            ax.set_xticks(range(3),names,rotation=60,ha='right',fontsize=7);self._reference_ticks(ax,order)
            ax.yaxis.set_major_formatter(FuncFormatter(fmt));ax.tick_params(axis='y',labelsize=6.7,pad=2);ax.grid(axis='y',color='#e9e9e9',lw=.35);ax.set_axisbelow(True)
            ax._eval_heading='Base'
            ax.set_ylabel('Episode return',fontsize=6.5,labelpad=2)
            self._row_label(fig,ax,env,x=.012)
        self._place_headings(fig,axes,categories=True)
        legend=fig.legend(handles=[Patch(facecolor=COLORS[k],label=LABELS[k]) for k in order]+base_modified_handles(),loc='lower center',bbox_to_anchor=(.54,-.015),ncol=5,fontsize=6.7,handlelength=1.2,frameon=False)
        self._reference_legend(legend,order)
        self._check_row_spacing(fig,axes)
        self._check_legend_spacing(fig,axes,legend)
        self.emit(fig,stem)

    def table_image(self,stem,headers,rows,bold=(),source_blocks=False,col_widths=None,font_size=8,row_height=.17):
        display=[];source_rows=set();translated_bold=[]
        if source_blocks:
            headers=headers[1:]
            for i,row in enumerate(rows):
                if i%8==0:
                    source_rows.add(len(display)+1)
                    display.append(['Source: '+row[0]]+['']*(len(headers)-1))
                display.append(row[1:])
            translated_bold=[(r+2+r//8,c-1) for r,c in bold]
        else:
            display=rows
            translated_bold=[(r+1,c) for r,c in bold]
        height=max(1.6,row_height*len(display)+.40)
        fig,ax=plt.subplots(figsize=(PAPER_WIDTH,height));ax.axis('off')
        table=ax.table(cellText=display,colLabels=headers,cellLoc='right',colLoc='center',bbox=[0,.02,1,.96],colWidths=col_widths)
        table.auto_set_font_size(False);table.set_fontsize(font_size)
        for (row,col),cell in table.get_celld().items():
            cell.visible_edges='';cell.set_facecolor('white');cell.set_edgecolor('#555555');cell.set_linewidth(.45)
            if row==0:
                cell.visible_edges='BT';cell.get_text().set_fontweight('semibold')
            if row==len(display):cell.visible_edges+='B'
            if col==0:cell.get_text().set_ha('left')
            if row in source_rows:
                cell.get_text().set_fontstyle('italic');cell.get_text().set_fontsize(7.5)
        for row,col in translated_bold:table[row,col].get_text().set_fontweight('bold')
        fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.01)
        self.emit(fig,stem)
