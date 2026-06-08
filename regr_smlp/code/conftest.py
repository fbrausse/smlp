import subprocess, os, shlex

if os.environ.get("DRYRUN"):
    def _dry_run(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            cmd = list(cmd)
            if '-out_dir' in cmd:
                cmd[cmd.index('-out_dir') + 1] = '.'
            print("DRYRUN:", ' '.join(shlex.quote(a) for a in cmd))
        else:
            print("DRYRUN:", cmd)
        return subprocess.CompletedProcess(cmd, 0)
    subprocess.run = _dry_run
