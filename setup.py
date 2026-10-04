#!/usr/bin/env python
"""Installation script."""
# import imp  # inline
# import importlib  # inline
import shlex as _sh
import os
import setuptools as _stp
import subprocess
import sys


NAME = 'polytope'
VERSION_FILE = f'{NAME}/_version.py'
MAJOR = 0
MINOR = 2
MICRO = 6
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'
VERSION_TEXT = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n")


def git_version(
        version:
            str
        ) -> str:
    """Return version with local version identifier."""
    sha = subprocess.check_output([
        'git', 'log', '-1', '--format=%H'
    ], text=True).strip()
    status_output = subprocess.check_output([
        'git', 'status', '--porcelain', '--untracked-files=normal'
    ], text=True).strip()
    if status_output:
        return f'{version}.dev0+{sha}.dirty'
    # commit is clean
    # is it release of `version` ?
    try:
        tag = subprocess.check_output([
            'git', 'describe', '--match=v[0-9]*', '--exact-match', '--tags', '--dirty'
        ], text=True).strip()
    except subprocess.CalledProcessError:
        return f'{version}.dev0+{sha}'
    assert tag == f'v{version}', (tag, version)
    return version


def run_setup():
    """Get version from git, then install."""
    try:
        version = git_version(VERSION)
    except AssertionError:
        raise
    except Exception:
        print('No git info: Assume release.')
        version = VERSION
    s = VERSION_TEXT.format(version=version)
    with open(VERSION_FILE, 'w') as f:
        f.write(s)
    _stp.setup(
        version=version)


if __name__ == '__main__':
    run_setup()
