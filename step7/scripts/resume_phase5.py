"""Sequential resume driver for long local experiments and dependent outputs.

Use one driver at a time. A saved process ID/stage makes unattended work auditable;
failure stops the pipeline instead of publishing partial research as complete.
"""
import argparse
from datetime import datetime,timezone
import os
import subprocess
import sys
import time

from step7.scripts.common import ROOT,STEP7,read_json,write_json


def state(status,stage,**details):
    write_json(STEP7/"analysis/pipeline_state.json",{"status":status,"stage":stage,"pid":os.getpid(),"updated_utc":datetime.now(timezone.utc).isoformat(),**details})


def wait_for_process(pid):
    """Wait via Windows synchronization handle; never send a signal to a process."""
    if os.name!="nt":
        raise ValueError("--wait-for-pid is only supported on this Windows host")
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:
        return
    try:
        while kernel.WaitForSingleObject(handle,30000)==0x00000102:
            state("running","waiting_for_existing_matrix",waiting_for_pid=pid)
    finally:
        kernel.CloseHandle(handle)


def run_module(module,*arguments):
    state("running",module)
    print(f"stage: {module}",flush=True)
    result=subprocess.run([sys.executable,"-m",module,*arguments],cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"stage failed: {module}; exit={result.returncode}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait-for-pid",type=int)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    try:
        if args.wait_for_pid:
            state("running","waiting_for_existing_matrix",waiting_for_pid=args.wait_for_pid)
            wait_for_process(args.wait_for_pid)
        if (STEP7/"config/l2x_semantic_amendment.json").exists():
            # Do not overwrite corrected sensitivity evidence with the original runner.
            run_module("step7.scripts.amend_l2x")
        else:
            run_module("step7.scripts.run_matrix","--collect-only")
            progress=read_json(STEP7/"analysis/matrix_progress.json")
            if progress["status"]!="complete":
                run_module("step7.scripts.run_matrix","--workers",str(args.workers))
        run_module("step7.scripts.analyze_results")
        run_module("step7.scripts.fit_candidates")
        run_module("step7.scripts.run_stress")
        run_module("step7.scripts.build_delivery")
        state("running","step7_tests")
        result=subprocess.run([sys.executable,"-m","unittest","discover","-s","step7/tests","-v"],cwd=ROOT)
        if result.returncode:
            raise RuntimeError("Step7 test suite failed")
        run_module("step7.scripts.audit_feature_dictionary")
        run_module("step7.scripts.validate_phase50")
        state("needs_review","independent_review_and_visual_QA",note="Numerical pipeline is complete; agent must inspect figures/report, independently review and resolve important findings before final completion.")
        print("pipeline complete; independent review/visual QA remain",flush=True)
    except Exception as exc:
        state("failed","repair_and_resume",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__=="__main__":
    main()
