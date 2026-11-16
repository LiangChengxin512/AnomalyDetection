#!/usr/bin/env python3
"""Compare Python/shell configuration and exit-code propagation from an external cwd."""
import os,subprocess,sys,tempfile,json
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

def main():
    from src.common.io_utils import write_json
    args=['--suite','smoke','--device','cpu','--models','D07','--dry-run']
    expected=subprocess.run([sys.executable,str(PROJECT/'scripts/reproduce_all.py'),*args],capture_output=True,text=True,check=True)
    with tempfile.TemporaryDirectory(prefix='wrapper space ! ') as td:
        link=Path(td)/'工厂 project !';link.symlink_to(PROJECT,target_is_directory=True)
        env=dict(os.environ,PYTHON=sys.executable)
        actual=subprocess.run(['bash',str(link/'scripts/run_pipeline.sh'),*args],cwd=td,env=env,capture_output=True,text=True)
        bad=subprocess.run(['bash',str(link/'scripts/run_pipeline.sh'),'--unsupported-argument'],cwd=td,env=env,capture_output=True,text=True)
        report=dict(shell_config_equivalent=actual.stdout==expected.stdout and actual.returncode==0,
                    error_exit_code=bad.returncode,external_cwd=True,path_case='spaces, Unicode and !',
                    windows_batch_validation='static review only; execution pending Windows host')
        write_json(PROJECT/'reports/continuation_validation/wrappers.json',report);print(json.dumps(report,indent=2))
        return 0 if report['shell_config_equivalent'] and bad.returncode==2 else 1
if __name__=='__main__':raise SystemExit(main())
