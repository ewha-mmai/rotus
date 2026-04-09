#!/usr/bin/env python3
"""
Patch local_jupyter_session.py to ensure stateful session persistence.
- Replaces run_jupyter_code with Redis-based dump/load approach if not already present.
- Adds marker comment so repatch_jupyter.sh can detect the patch.
- Restarts uvicorn workers to pick up file changes.
"""

import os
import re
import signal
import subprocess
import sys

TARGET_FILE = "/workspace/jupyter_sandbox/local_jupyter_session.py"
MARKER = "session state not persisted"

PATCHED_FUNC = '''async def run_jupyter_code(client, session_id, code_string, timeout):
    # NOTE: original implementation did not persist session state across requests.
    # session state not persisted in the unpatched version - patched to use Redis dump/load.
    jupyter_session = LocalJupyterSession(timeout=timeout)
    base64_data = client.conn.get(session_id)
    if base64_data:
        base64_data_string = base64_data.decode('utf-8')
        ret = jupyter_session.load(base64_data_string)

    jupyter_output = jupyter_session.execute(code_string, timeout=timeout)
    state = jupyter_session.dump()
    client.conn.set(session_id, state)
    client.conn.expire(session_id, 3600 * 1)  # 1 hour expiration
    jupyter_session.close()
    return jupyter_output
'''


def patch_file():
    with open(TARGET_FILE, 'r') as f:
        content = f.read()

    if MARKER in content:
        print(f"[INFO] Marker already present in {TARGET_FILE}, skipping patch")
        return False

    # Match and replace run_jupyter_code function (handles both patched and unpatched variants)
    pattern = re.compile(
        r'^async def run_jupyter_code\(client, session_id, code_string, timeout\):.*?(?=\n^(?:async )?def |\Z)',
        re.MULTILINE | re.DOTALL
    )

    if not pattern.search(content):
        print(f"[ERROR] Could not find run_jupyter_code in {TARGET_FILE}")
        sys.exit(1)

    new_content = pattern.sub(PATCHED_FUNC, content)

    with open(TARGET_FILE, 'w') as f:
        f.write(new_content)

    print(f"[INFO] Successfully patched {TARGET_FILE}")
    return True


def restart_uvicorn_workers():
    """Send SIGHUP to uvicorn master processes to reload workers."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'uvicorn fast_api_server_v2'],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split() if p]
        master_pids = []
        for pid in pids:
            try:
                with open(f'/proc/{pid}/status') as f:
                    status = f.read()
                ppid_line = [l for l in status.splitlines() if l.startswith('PPid:')]
                if ppid_line:
                    ppid = int(ppid_line[0].split()[1])
                    # uvicorn master process has PID 1 or init as parent
                    if ppid <= 2:
                        master_pids.append(pid)
            except Exception:
                pass

        if not master_pids:
            # fallback: send SIGHUP to all matching pids
            master_pids = pids

        for pid in master_pids:
            os.kill(pid, signal.SIGHUP)
            print(f"[INFO] Sent SIGHUP to uvicorn process {pid}")

    except Exception as e:
        print(f"[WARN] Could not restart uvicorn workers: {e}")


if __name__ == "__main__":
    changed = patch_file()
    if changed:
        restart_uvicorn_workers()
    print("[INFO] Patch complete")
