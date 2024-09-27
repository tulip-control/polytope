#!/usr/bin/env python
"""Installation script."""
# import imp  # inline
# import importlib  # inline
import shlex as _sh
import os
import setuptools as _stp
import subprocess
import sys


classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']


def retrieve_git_info():
    """Return commit hash of HEAD, or "release", or None if failure.

    If the git command fails, then return None.

    If HEAD has tag with prefix "vM" where M is an integer, then
    return 'release'.
    Tags with such names are regarded as version or release tags.

    Otherwise, return the commit hash as str.
    """
    # Is Git installed?
    cmd = _sh.split('''
        git --version
        ''')
    try:
        subprocess.call(
            cmd,
            stdout=subprocess.PIPE)
    except OSError:
        return None
    # Decide whether this is a release
    cmd = _sh.split('''
        git describe
            --tags
            --candidates=0
                HEAD
        ''')
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode == 0:
        tag = p.stdout.read().decode('utf-8')
        if len(tag) >= 2 and tag.startswith('v'):
            try:
                int(tag[1])
                return 'release'
            except ValueError:
                pass
    # Otherwise, return commit hash
    cmd = _sh.split('''
        git log
            -1
            --format=%H
        ''')
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE)
    p.wait()
    sha1 = p.stdout.read().decode('utf-8')
    return sha1


def run_setup():
    """Get version from git, then install."""
    # load long description from README.rst
    readme_file = 'README.rst'
    if os.path.exists(readme_file):
        with open(readme_file) as f:
            long_description = f.read()
    else:
        print(
            'Could not find readme file to '
            'extract long_description.')
        long_description = ''
    # If .git directory is present,
    # create commit_hash.txt accordingly
    # to indicate version information
    if os.path.exists('.git'):
        # Provide commit hash or
        # empty file to indicate release
        sha1 = retrieve_git_info()
        if sha1 is None:
            sha1 = 'unknown-commit'
        elif sha1 == 'release':
            sha1 = ''
        commit_hash_header = (
            '# DO NOT EDIT!  '
            'This file was automatically generated by '
            'setup.py of polytope')
        with open('polytope/commit_hash.txt', 'w') as f:
            f.write(commit_hash_header + '\n')
            f.write(sha1 + '\n')
    # Import polytope/version.py
    # without importing polytope
    if sys.version_info.major == 2:
        import imp
        version = imp.load_module(
            'version',
            *imp.find_module('version', ['polytope']))
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'version', 'polytope/version.py')
        version = importlib.util.module_from_spec(spec)
        sys.modules['version'] = version
        spec.loader.exec_module(version)
    polytope_version = version.version
    _stp.setup(
        name='polytope',
        version=polytope_version,
        description='Polytope Toolbox',
        long_description=long_description,
        author='Caltech Control and Dynamical Systems',
        author_email='polytope@tulip-control.org',
        url='https://tulip-control.org',
        project_urls={
            'Bug Tracker':
                'https://github.com/tulip-control/polytope/issues',
            'Documentation':
                'https://tulip-control.github.io/polytope/',
            'Source Code':
                'https://github.com/tulip-control/polytope'},
        license='BSD',
        python_requires='>=3.8',
        setup_requires=[
            'setuptools >= 65.5.1'],
        install_requires=[
            'networkx >= 3.0',
            'numpy >= 1.24.1',
            'scipy >= 1.10.0'],
        tests_require=[
            'matplotlib >= 3.6.3',
            'pytest >= 7.2.1'],
        packages=[
            'polytope'],
        package_dir=dict(
            polytope='polytope'),
        package_data=dict(
            polytope=['commit_hash.txt']),
        classifiers=classifiers)


if __name__ == '__main__':
    run_setup()
