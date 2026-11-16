"""Manifest-driven dependency tasks with verified output reuse and failure isolation."""
import time
import traceback
import inspect
import shutil
import pandas as pd
from src.common.io_utils import read_json,write_json
from src.common.provenance import digest,inventory,now


class Tasks:
    def __init__(self,root,resume=False,fail_fast=False):
        self.root,self.resume,self.fail_fast=root,resume,fail_fast
        self.rows=[];self.status={};self.outputs={}
        self.root.mkdir(parents=True,exist_ok=True)
    def flush(self):
        pd.DataFrame(self.rows).to_csv(self.root/'task_registry.csv',index=False)
    def run(self,task_id,dependencies,relative_output,fingerprint,fn):
        output=self.root/relative_output
        row=dict(schema_version='2.0',task_id=task_id,dependencies=','.join(dependencies),status='PENDING',
                 reason='',started_at=None,finished_at=None,seconds=None,exit_code=None,artifacts=relative_output.as_posix())
        self.rows.append(row);self.flush()
        if any(self.status.get(d) not in ['SUCCEEDED','REUSED','NOT_APPLICABLE'] for d in dependencies):
            row.update(status='SKIPPED_DEPENDENCY',reason='A required upstream task failed',exit_code=1)
        else:
            key=digest(dict(config=fingerprint,function_source=inspect.getsource(fn),
                            dependencies={d:self.outputs[d] for d in dependencies}))
            receipt=self.root/'receipts'/f'{task_id}.json'
            if self.resume and receipt.exists():
                old=read_json(receipt)
                if old['fingerprint']==key and output.exists() and inventory(output)==old['files']:
                    row.update(status='REUSED',reason='Input and output hashes verified',exit_code=0)
                    self.outputs[task_id]=digest(old['files'])
                    self.status[task_id]=row['status'];self.flush();return output
            row.update(status='RUNNING',started_at=now());self.flush();began=time.perf_counter()
            print(f'[{task_id}] running -> {relative_output}',flush=True)
            try:
                # Only training needs an incomplete epoch-boundary checkpoint.
                # Other invalidated stages must not retain stale curves/partitions.
                if output.exists() and not task_id.startswith('fit_'):
                    if not output.resolve().is_relative_to(self.root.resolve()) or output.resolve()==self.root.resolve():
                        raise ValueError('Task output must be below its run root')
                    shutil.rmtree(output)
                output.mkdir(parents=True,exist_ok=True)
                fn(output)
                (output/'failure.txt').unlink(missing_ok=True)
                files=inventory(output)
                if not files:raise ValueError('Task produced no artifacts')
                write_json(receipt,dict(fingerprint=key,files=files))
                self.outputs[task_id]=digest(files)
                row.update(status='SUCCEEDED',exit_code=0)
            except (Exception,KeyboardInterrupt) as exc:
                row.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}',exit_code=130 if isinstance(exc,KeyboardInterrupt) else 1)
                (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                print(f'[{task_id}] FAILED: {exc}',flush=True)
                if self.fail_fast or isinstance(exc,KeyboardInterrupt):
                    row.update(seconds=time.perf_counter()-began,finished_at=now());self.status[task_id]=row['status'];self.flush();raise
            row.update(seconds=time.perf_counter()-began,finished_at=now())
        self.status[task_id]=row['status'];self.outputs.setdefault(task_id,'failed');self.flush()
        return output
