#!/usr/bin/env python3
"""
Run auditwheel repair on the smlp wheel in dist/ to produce a
self-contained manylinux wheel with all .so dependencies bundled.

Usage:
    python3 repair_wheel.py [dist_dir] [--plat PLATFORM]

dist_dir defaults to 'dist/'.
--plat defaults to 'manylinux_2_28_x86_64'.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def _find_auditwheel_location() -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "auditwheel"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()

    user_site = (
        Path.home() / ".local" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if (user_site / "auditwheel").exists():
        return str(user_site)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", default="dist")
    parser.add_argument("--plat", default="manylinux_2_28_x86_64",
                        help="Target platform tag (default: manylinux_2_28_x86_64)")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir)

    location = _find_auditwheel_location()
    if not location:
        print("ERROR: auditwheel not found.")
        print("Install with: python3 -m pip install auditwheel patchelf")
        sys.exit(1)

    wheels = sorted(dist_dir.glob("smlptech-*linux_x86_64.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print(f"ERROR: No linux_x86_64 wheel found in {dist_dir}/")
        sys.exit(1)

    wheel = wheels[-1]
    print(f"Repairing {wheel} with platform {args.plat} ...")
    subprocess.check_call([
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {location!r}); "
        f"from auditwheel.main import main; sys.exit(main())",
        "repair", str(wheel), "--plat", args.plat, "-w", str(dist_dir)
    ])
    print(f"Done. manylinux wheel saved to {dist_dir}/")


if __name__ == "__main__":
    main()
