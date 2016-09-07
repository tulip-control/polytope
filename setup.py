#!/usr/bin/env python
"""Installation script."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import imp
import os
from setuptools import setup

"""polytope package version"""
version_info = (0, 1, 4)

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
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']


def run_setup():
    """Get version from git, then install."""
    # load long description from README.rst
    readme_file = 'README.rst'
    if os.path.exists(readme_file):
        long_description = open(readme_file).read()
    else:
        print('Could not find readme file to extract long_description.')
        long_description = ''

    polytope_version = '.'.join([str(x) for x in version_info])

    # Import polytope/version.py without importing polytope
    version = imp.load_module('version',
                              *imp.find_module('version', ['polytope']))
    from version import append_dev_info
    polytope_version = append_dev_info(polytope_version)

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
        requires=['numpy', 'scipy', 'networkx'],
        install_requires=[
            'numpy >= 1.6',
            'scipy >= 0.16',
            'networkx >= 1.6'],
        tests_require=[
            'nose',
            'matplotlib'],
        packages=[
            'polytope'],
        package_dir=dict(polytope='polytope'),
        package_data=dict(polytope=['commit_hash.txt']),
        classifiers=classifiers)


if __name__ == '__main__':
    run_setup()
