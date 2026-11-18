"""Portable CSV monitor; historical replay is an advisory workflow, not process control."""
from __future__ import annotations
import argparse
import queue
import shlex
import sys
import threading
import time
from pathlib import Path
from src.common.config import PROJECT_ROOT
from src.common.io_utils import sha256_file
from src.common.provenance import run_lock
from src.data.prepare_factory import load_site,read_telemetry,path_for
from src.runtime.bundle import load_bundle
from src.runtime.case_store import CaseStore


def display(update,engine,watched,position,total):
    d,a=update['detection'],update['alarm']
    score=f'{d["score"]:.5g}' if d['score'] is not None else 'unavailable'
    text=f'{position}/{total} {d["timestamp"]} score={score} threshold={d["threshold"]:.5g} state={a["state"]}'
    print(text,flush=True)
    if update['rankings']:
        r=update['rankings'][-1]
        print('  Candidates: '+', '.join(c['tag'] for c in r['candidates'][:3])+f' | evidence margin={r["margin"]:.4g}')
    if update['advice']:
        advice=update['advice'];print(f'  Advice: {advice["description_en"] or advice["reason"]} [{advice["mode"]}]')
    if a['alarm_id']:print(f'  Case: {a["alarm_id"]}')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',required=True);ap.add_argument('--site');ap.add_argument('--input',required=True)
    ap.add_argument('--device',choices=['auto','cpu','mps','cuda']);ap.add_argument('--speed',type=float,default=1)
    ap.add_argument('--step',action='store_true');ap.add_argument('--plain',action='store_true')
    ap.add_argument('--session-id',default=None);ap.add_argument('--max-rows',type=int,default=0)
    ap.add_argument('--display-every',type=int,default=1);ap.add_argument('--commands-file',help='JSONL local operator commands, applied after replay')
    args=ap.parse_args(argv)
    if args.speed<=0:ap.error('--speed must be positive')
    if args.max_rows<0 or args.display_every<1:ap.error('Row cap must be nonnegative and display cadence positive')
    site=load_site(args.site) if args.site else None
    manifest,engine=load_bundle(path_for(args.bundle),args.device,site)
    site=site or engine.site
    source=path_for(args.input)
    frame=read_telemetry(source,site)
    if args.max_rows:frame=frame.iloc[:args.max_rows]
    sid=args.session_id or time.strftime('session_%Y%m%d_%H%M%S')
    if Path(sid).name!=sid or sid in ['.','..']:ap.error('Session ID must be one directory name')
    session=PROJECT_ROOT/'app_runs'/sid
    print(f'Site={site["site_id"]} bundle={manifest["bundle_id"]} device={engine.detector.device} scope={manifest["validation_scope"]}')
    print('Capabilities: '+', '.join(manifest['capabilities']))
    print('Commands: pause, resume, step, speed N, watch TAG, acknowledge CASE REASON, reject-advice CASE REASON, escalate CASE REASON, add-note CASE REASON, close-case CASE REASON, export, quit')
    inbox=queue.Queue()
    if sys.stdin.isatty():
        def reader():
            for line in sys.stdin:inbox.put(line.strip())
        threading.Thread(target=reader,daemon=True).start()
    paused=args.step;single=False;speed=args.speed;watched=[];stop=False
    with run_lock(session/'.lock'):
        store=CaseStore(session)
        store.bind(dict(bundle_id=manifest['bundle_id'],manifest_hash=sha256_file(path_for(args.bundle)/'bundle_manifest.json'),source_sha256=sha256_file(source)))
        def command(line):
            nonlocal paused,single,speed,stop
            tokens=shlex.split(line)
            if not tokens:return
            name=tokens[0]
            if name=='pause':paused=True
            elif name in ['resume','start']:paused=False
            elif name=='step':single=True;paused=True
            elif name=='speed':
                value=float(tokens[1])
                if value<=0:raise ValueError('Speed must be positive')
                speed=value
            elif name=='watch':
                if tokens[1] not in engine.tags:raise ValueError('Unknown tag')
                watched.append(tokens[1])
            elif name=='export':print(f'Exported {store.export()}')
            elif name in ['quit','exit']:stop=True
            else:store.command(tokens[1],name,'operator',' '.join(tokens[2:]))
        try:
            for i,values in enumerate(frame.itertuples(index=False,name=None)):
                row=dict(zip(frame.columns,values))
                # Rebuild causal state from the prefix; DB uniqueness prevents replay duplicates.
                seen=store.contains(row['sample_id'])
                while not seen:
                    while not inbox.empty():
                        try:command(inbox.get_nowait())
                        except (ValueError,IndexError) as exc:print(f'Command rejected: {exc}')
                    if stop or not paused or single:break
                    if not sys.stdin.isatty():raise ValueError('--step requires an interactive terminal')
                    time.sleep(.05)
                if stop:break
                engine.cache.clear()
                update=engine.process_frame(row)
                if not seen:
                    store.apply_update(update)
                    if i%args.display_every==0 or update['alarm']['opened'] or update['advice']:
                        display(update,engine,watched,i+1,len(frame))
                        for tag in watched:
                            feature=next(f for f in site['features'] if f['name']==tag)
                            print(f'  {tag}={row[tag]} {feature.get("unit","unknown")} quality={row["quality_flags"] or "valid"}')
                    single=False
                    # Event time is unaffected by playback speed. Never skip sensor rows.
                    time.sleep(site['input']['sample_interval_seconds']/speed)
            if args.commands_file:
                import json
                for line in path_for(args.commands_file).read_text(encoding='utf-8').splitlines():
                    cmd=json.loads(line);store.command(cmd['case_id'],cmd['command'],cmd['actor'],cmd['reason'])
            print('Input ended: DATA_QUALITY_HOLD. End of input is not evidence of recovery.')
        finally:
            store.close()
    print(f'Session exports: {session}')
    return 0
if __name__=='__main__':raise SystemExit(main())
