#!/usr/bin/env python3
"""Redraw only frozen CSV inputs; this module never imports training or calibration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def render(run,plot_config,figure=None,export_data_only=False):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(run/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import yaml
    from src.common.io_utils import write_json,sha256_file,read_json
    from src.common.provenance import source_hash
    config=yaml.safe_load(Path(plot_config).read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size':config.get('font_size',10)})
    scope=read_json(run/'run_manifest.json')['scope'].upper()
    output=run/'figures';output.mkdir(exist_ok=True)
    records=[]
    for fid,spec in config['figures'].items():
        if figure and fid!=figure:continue
        if spec.get('kind') in ['curve','timeline','sensitivity','coverage','effect']:
            records.append(render_detail(run,output,fid,spec,config,scope,export_data_only))
            continue
        source=run/'plot_data'/spec['input']
        if not source.exists():continue
        df=pd.read_csv(source)
        selection=df.loc[df.metric.eq(spec['metric'])&df.status.eq('SUCCEEDED')] if 'metric' in df else df.iloc[:0]
        for col,value in spec.get('filters',{}).items():selection=selection.loc[selection[col].eq(value)]
        record=dict(figure_id=fid,input_csv=str(source.relative_to(run)),input_sha256=sha256_file(source),plot_config_hash=sha256_file(plot_config))
        if selection.empty:
            record.update(status='NOT_APPLICABLE',reason='No defined observations for this figure');records.append(record);continue
        if export_data_only:records.append(dict(record,status='DATA_ONLY'));continue
        # Separate target scopes and conditioning instead of silently averaging unlike protocols.
        facets=[c for c in ['target_scope','conditioning','snapshot_seconds'] if c in selection and selection[c].nunique()>1]
        groups=list(selection.groupby(facets,dropna=False)) if facets else [('all',selection)]
        for facet,part in groups:
            reduced=part.groupby(spec['x'],dropna=False).value.agg(['mean','std'])
            fig,ax=plt.subplots(figsize=config['figsize'])
            errors=reduced['std'].fillna(0) if spec.get('show_seed_std',False) else None
            ax.bar(reduced.index.astype(str),reduced['mean'],yerr=errors,capsize=3,color=spec.get('color',config['color']))
            if errors is not None:ax.text(.99,.99,'Error bars: seed standard deviation',ha='right',va='top',transform=ax.transAxes,fontsize=8)
            ax.set_ylabel(spec['metric']);ax.set_title(f'{scope} — {fid}: {facet}');fig.tight_layout()
            suffix='_'+str(facet).replace(' ','').replace("'",'').replace('(','').replace(')','').replace(',','_') if facets else ''
            for fmt in config['formats']:fig.savefig(output/f'{fid}{suffix}.{fmt}',dpi=config['dpi'])
            plt.close(fig)
        records.append(dict(record,status='SUCCEEDED'))
    history=run/'plot_data/training_history.csv'
    if history.exists() and (not figure or figure=='training') and not export_data_only:
        df=pd.read_csv(history)
        if len(df):
            groups=list(df.groupby(['model_id','seed']))
            columns=min(3,len(groups));rows=(len(groups)+columns-1)//columns
            fig,axes=plt.subplots(rows,columns,figsize=(config['figsize'][0],max(config['figsize'][1],3*rows)),squeeze=False)
            for ax,((name,seed),part) in zip(axes.flat,groups):
                ax.plot(part.epoch,part.train_recon_loss,label='Train reconstruction')
                ax.plot(part.epoch,part.val_recon_mse,label='Normal validation')
                best_rows=part.loc[part.improved_best.eq(True)] if 'improved_best' in part else part.iloc[:0]
                if len(best_rows):
                    best=best_rows.iloc[-1]
                    ax.scatter([best.epoch],[best.val_recon_mse],marker='*',s=70,color='#16a34a',zorder=5,label=f'Best epoch: {int(best.epoch)}')
                ax.axvline(part.epoch.max(),color='#64748b',linestyle='--',label='Last completed epoch')
                ax.set(xlabel='Epoch',ylabel='Reconstruction loss',title=f'{name} / seed {seed}')
                ax.legend(fontsize=7)
            for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
            fig.suptitle(f'{scope} — training');fig.tight_layout()
            for fmt in config['formats']:fig.savefig(output/f'training.{fmt}',dpi=config['dpi'])
            plt.close(fig)
            records.append(dict(figure_id='training',input_csv=str(history.relative_to(run)),input_sha256=sha256_file(history),status='SUCCEEDED'))
    write_json(output/'figure_manifest.json',dict(figures=records,plot_config_hash=sha256_file(plot_config),code_hash=source_hash([Path(__file__).parent])))
    return records


def render_detail(run,output,fid,spec,config,scope,export_data_only):
    """Render already-exported measurements; unavailable risk stays unavailable."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.common.io_utils import sha256_file
    sources=sorted((run/'plot_data').glob(spec['input']))
    record=dict(figure_id=fid,input_hashes={str(p.relative_to(run)):sha256_file(p) for p in sources})
    frames=[pd.read_csv(p) for p in sources]
    frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    for column,value in spec.get('filters',{}).items():
        if column in frame:frame=frame.loc[frame[column].eq(value)]
    kind=spec['kind']
    required=['timestamp','score','threshold'] if kind=='timeline' else [spec.get('x'),spec.get('y')]
    available=len(frame)>0 and all(c in frame for c in required)
    if available and kind!='timeline':available=frame[spec['y']].notna().any()
    if not available:
        reason='No defined observations in the exported CSV inputs'
        if kind=='effect':reason='Physical action response is unqualified; surrogate risk is unavailable'
        (output/f'{fid}_unavailable.md').write_text(f'# {scope}: {fid}\n\n{reason}. No zero-valued effect is plotted.\n',encoding='utf-8')
        return dict(record,status='NOT_APPLICABLE',reason=reason)
    if export_data_only:return dict(record,status='DATA_ONLY')
    if kind=='timeline':
        groups=list(frame.groupby(['source_file','segment_id'],sort=False))
        fig,axes=plt.subplots(len(groups),1,figsize=(config['figsize'][0],max(config['figsize'][1],2.8*len(groups))),squeeze=False)
        for ax,((source,segment),part) in zip(axes.flat,groups):
            timestamps=pd.to_datetime(part.timestamp)
            ax.plot(timestamps,part.score,label='Score',linewidth=.8)
            ax.plot(timestamps,part.threshold,label='Locked threshold',linestyle='--',linewidth=.9)
            if 'label' in part:
                ax.fill_between(timestamps,0,1,where=part.label.to_numpy(bool),transform=ax.get_xaxis_transform(),color='#ef4444',alpha=.13,label='Evaluation attack label')
            if 'alarm_state' in part:
                opened=part.alarm_state.eq('ACTIVE') & ~part.alarm_state.shift().eq('ACTIVE')
                ax.scatter(timestamps[opened],part.loc[opened,'score'],marker='^',s=14,color='#f59e0b',label='Alarm activation')
            ax.set(title=f'{source} / segment {segment}',ylabel='Anomaly score',xlabel='Source time')
            ax.legend(fontsize=7)
        fig.suptitle(f'{scope} — {fid} (evaluation only)');fig.autofmt_xdate();fig.tight_layout()
    else:
        fig,ax=plt.subplots(figsize=config['figsize'])
        group_columns=[c for c in spec.get('group_by',[]) if c in frame]
        group_key=group_columns[0] if len(group_columns)==1 else group_columns
        groups=list(frame.groupby(group_key,dropna=False,sort=False)) if group_columns else [('all',frame)]
        color_keys=list(dict.fromkeys((k[0] if isinstance(k,tuple) else k) for k,_ in groups))
        colors=plt.get_cmap('tab10')
        for key,part in groups:
            part=part.dropna(subset=[spec['x'],spec['y']])
            if not len(part):continue
            if kind!='curve':part=part.sort_values(spec['x'])
            values=key if isinstance(key,tuple) else (key,)
            label=' / '.join('deterministic' if pd.isna(v) else str(v) for v in values)
            if group_columns==['seed']:label='seed '+label
            color=colors(color_keys.index(values[0])%10)
            seed_style={1:'-',7:'--',21:':'}
            style=seed_style.get(values[-1],'-') if len(values)>1 else '-'
            ax.plot(part[spec['x']],part[spec['y']],label=label,color=color,linestyle=style,marker='o' if kind in ['sensitivity','coverage'] else None,markersize=3,linewidth=1)
            if kind=='coverage' and 'specific_coverage' in part:
                ax.plot(part[spec['x']],part.specific_coverage,color=color,linestyle='--',marker='x',markersize=3,label=f'{label} specific')
        ax.set(xlabel=spec.get('xlabel',spec['x']),ylabel=spec.get('ylabel',spec['y']),title=f'{scope} — {fid}')
        if kind=='coverage':ax.set_ylim(-.02,1.02)
        if kind=='curve':ax.set(xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=7,ncol=2 if len(groups)>6 else 1)
        ax.grid(alpha=.15);fig.tight_layout()
    files=[]
    for fmt in config['formats']:
        path=output/f'{fid}.{fmt}';fig.savefig(path,dpi=config['dpi']);files.append(str(path.relative_to(run)))
    plt.close(fig)
    (output/f'{fid}_unavailable.md').unlink(missing_ok=True)
    return dict(record,status='SUCCEEDED',outputs=files)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--plot-config',default='config/plots.yaml')
    ap.add_argument('--figure');ap.add_argument('--export-data-only',action='store_true');args=ap.parse_args()
    from src.data.prepare_factory import path_for
    render(path_for(args.run_dir),path_for(args.plot_config),args.figure,args.export_data_only)
if __name__=='__main__':main()
