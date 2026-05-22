#!/usr/bin/env python3.11
import requests
from packaging.version import Version
from argparse import ArgumentParser, Namespace
from os.path import realpath, basename
from sys import argv

def add_arguments() -> ArgumentParser:
    p = ArgumentParser()
    p.add_argument('--test', '-t', default=None, action='store_true')
    p.add_argument('--all',  '-a', default=None, action='store_true')
    p.add_argument('--latest',  '-l', default=None, action='store_true')
    return p

def get_release_files(package: str, version: str, test_version: bool = False) -> list:
    url = f"https://test.pypi.org/pypi/{package}/{version}/json" if test_version \
                                                                else f"https://pypi.org/pypi/{package}/{version}/json" 
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["urls"]  # list of file dicts

def smlp_versions(args: Namespace) -> int:
    script_name = basename(realpath(argv[0]))
    try:
        url = "https://test.pypi.org/pypi/smlptech/json" if args.test \
         else "https://pypi.org/pypi/smlptech/json"
        r = requests.get(url)
        data = r.json()
        version_list = list(data["releases"].keys())
        versions = sorted(Version(v) for v in data["releases"].keys())
        selected_versions = versions if args.all else [v for v in versions if not v.is_prerelease]
        if args.latest:
            print(selected_versions[-1])
        else:
            for v in selected_versions:
                if len(get_release_files("smlptech", v, args.test)) > 0:
                    print(v)

    except Exception as err:
        print(f"\n{script_name}: ERROR: {err}\n")
        return 1
    return 0

if __name__ == '__main__':
    exit(smlp_versions(add_arguments().parse_args()))
