import subprocess, os, shlex
import lib

if os.environ.get("DRYRUN"):
    _real_run = subprocess.run
    def _dry_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[0].endswith('smlp'):
            cmd = list(cmd)
            if '-out_dir' in cmd:
                cmd[cmd.index('-out_dir') + 1] = '.'
            print("DRYRUN:", ' '.join(shlex.quote(a) for a in cmd))
            return subprocess.CompletedProcess(cmd, 0)
        return _real_run(cmd, *args, **kwargs)
    lib.subprocess.run = _dry_run
