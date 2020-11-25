#!/usr/bin/env python
"""Installation script."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# import imp  # inline
# import importlib  # inline
import os
from setuptools import setup
import subprocess
import sys


classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
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
    try:
        subprocess.call(['git', '--version'],
                        stdout=subprocess.PIPE)
    except OSError:
        return None
    # Decide whether this is a release
    p = subprocess.Popen(
        ['git', 'describe', '--tags', '--candidates=0', 'HEAD'],
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
    p = subprocess.Popen(
        ['git', 'log', '-1', '--format=%H'],
        stdout=subprocess.PIPE)
    p.wait()
    sha1 = p.stdout.read().decode('utf-8')
    return sha1


def run_setup():
    """Get version from git, then install."""
    # load long description from README.rst
    readme_file = 'README.rst'
    if os.path.exists(readme_file):
        long_description = open(readme_file).read()
    else:
        print('Could not find readme file to extract long_description.')
        long_description = ''
    # If .git directory is present, create commit_hash.txt accordingly
    # to indicate version information
    if os.path.exists('.git'):
        # Provide commit hash or empty file to indicate release
        sha1 = retrieve_git_info()
        if sha1 is None:
            sha1 = 'unknown-commit'
        elif sha1 == 'release':
            sha1 = ''
        commit_hash_header = (
            '# DO NOT EDIT!  '
            'This file was automatically generated by setup.py of polytope')
        with open('polytope/commit_hash.txt', 'w') as f:
            f.write(commit_hash_header + '\n')
            f.write(sha1 + '\n')
    # Import polytope/version.py without importing polytope
    if sys.version_info.major == 2:
        import imp
        version = imp.load_module('version',
                                  *imp.find_module('version', ['polytope']))
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'version', 'polytope/version.py')
        version = importlib.util.module_from_spec(spec)
        sys.modules['version'] = version
        spec.loader.exec_module(version)
    polytope_version = version.version
    setup(
        name='polytope',
        version=polytope_version,
        description='Polytope Toolbox',
        long_description=long_description,
        author='Caltech Control and Dynamical Systems',
        author_email='polytope@tulip-control.org',
        url='http://tulip-control.org',
        bugtrack_url='http://github.com/tulip-control/polytope/issues',
        license='BSD',
        setup_requires=['setuptools >= 23.0.0'],
        install_requires=[
            'numpy >= 1.10.0',
            'scipy >= 0.18.0',
            'networkx >= 1.6'],
        tests_require=[
            'nose',
            'matplotlib >= 2.0.0'],
        packages=[
            'polytope'],
        package_dir=dict(polytope='polytope'),
        package_data=dict(polytope=['commit_hash.txt']),
        classifiers=classifiers)


if __name__ == '__main__':
    run_setup()
