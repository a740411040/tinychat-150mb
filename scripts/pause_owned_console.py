"""Send Ctrl+C only when the target console contains explicitly allowed PIDs."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--target-pid", type=int, required=True)
p.add_argument("--allowed-pid", type=int, action="append", required=True)
args = p.parse_args()
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.AttachConsole.argtypes = [wintypes.DWORD]
k.AttachConsole.restype = wintypes.BOOL
k.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
k.GetConsoleProcessList.restype = wintypes.DWORD
k.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
k.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
k.FreeConsole()
if not k.AttachConsole(args.target_pid):
    raise ctypes.WinError(ctypes.get_last_error())
try:
    if not k.SetConsoleCtrlHandler(None, True):
        raise ctypes.WinError(ctypes.get_last_error())
    ids = (wintypes.DWORD * 64)()
    count = k.GetConsoleProcessList(ids, 64)
    if not count or count > 64:
        raise RuntimeError("Cannot bound the target console process list")
    observed = set(ids[:count])
    allowed = set(args.allowed_pid) | {os.getpid()}
    if args.target_pid not in observed or observed - allowed:
        raise RuntimeError(f"Refusing an unrelated/shared console: {sorted(observed)}")
    if not k.GenerateConsoleCtrlEvent(0, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    print(json.dumps({"ctrl_c_sent": True, "console_pids": sorted(observed)}), flush=True)
finally:
    k.FreeConsole()
