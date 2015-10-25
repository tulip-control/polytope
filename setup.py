#!/usr/bin/env python

# for testing on https://testpypi.python.org use:
#
# pip install -i https://testpypi.python.org/pypi polytope
#             --extra-index-url https://pypi.python.org/pypi
#
# so that dependencies can be found and installed,
# when testing in a clean virtualenv.
import imp
import os
from setuptools import setup
import subprocess
import sys


###########################################
# Dependency or optional-checking functions
###########################################
# (see notes below.)
def check_glpk():
    try:
        import cvxopt.glpk
    except ImportError:
        return False
    return True


def check_mpl():
    try:
        import matplotlib
    except ImportError:
        return False
    return True

# Handle "dry-check" argument to check for dependencies without
# installing the polytope package; checking occurs by default if
# "install" is given, unless both "install" and "nocheck" are given
# (but typical users do not need "nocheck").

# You *must* have these to use the polytope package.  Each item in
# other_depends must be treated specially; thus other_depends is a
# dictionary with
#
#   keys   : names of dependency;

#   values : list of callable and string, which is printed on failure
#           (i.e. package not found); we interpret the return value
#           True to be success, and False failure.
other_depends = dict()
glpk_msg = (
    'GLPK seems to be missing\n'
    'and thus apparently not used by your installation of CVXOPT.\n'
    'If you\'re interested, see http://www.gnu.org/s/glpk/')
mpl_msg = (
    'matplotlib not found.\n'
    'For many graphics drawing features, you must install\n'
    'matplotlib (http://matplotlib.org/).')

# These are nice to have but not necessary. Each item is of the form
#
#   keys   : name of optional package;
#   values : list of callable and two strings, first string printed on
#           success, second printed on failure (i.e. package not
#           found); we interpret the return value True to be success,
#           and False failure.
optionals = dict(
    glpk=[check_glpk, 'GLPK found.', glpk_msg],
    matplotlib=[check_mpl, 'matplotlib found.', mpl_msg])


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
        tag = p.stdout.read()
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
    sha1 = p.stdout.read()
    return sha1


perform_setup = True
check_deps = False
if 'install' in sys.argv[1:] and 'nocheck' not in sys.argv[1:]:
    check_deps = True
elif 'dry-check' in sys.argv[1:]:
    perform_setup = False
    check_deps = True
# Pull "dry-check" and "nocheck" from argument list, if present, to play
# nicely with Distutils setup.
try:
    sys.argv.remove('dry-check')
except ValueError:
    pass
try:
    sys.argv.remove('nocheck')
except ValueError:
    pass
if check_deps:
    if not perform_setup:
        print('Checking for required dependencies...')
        # Python package dependencies
        try:
            import numpy
        except:
            print('ERROR: NumPy not found.')
            raise
        try:
            import networkx
        except:
            print('ERROR: NetworkX not found.')
            raise
        try:
            import cvxopt
        except:
            print('ERROR: CVXOPT not found.')
            raise
        # Other dependencies
        for (dep_key, dep_val) in other_depends.items():
            if not dep_val[0]():
                print(dep_val[1])
                raise Exception('Failed dependency: ' + dep_key)
    # Optional stuff
    for (opt_key, opt_val) in optionals.items():
        print('Probing for optional ' + opt_key + '...')
        if opt_val[0]():
            print('\t' + opt_val[1])
        else:
            print('\t' + opt_val[2])
# load long description from README.rst
readme_file = 'README.rst'
if os.path.exists(readme_file):
    long_description = open(readme_file).read()
else:
    print('Could not find readme file to extract long_description.')
    long_description = ''
if perform_setup:
    # If .git directory is present, create commit_hash.txt accordingly
    # to indicate version information
    if os.path.exists('.git'):
        # Provide commit hash or empty file to indicate release
        sha1 = retrieve_git_info()
        if sha1 is None:
            sha1 = 'unknown-commit'
        elif sha1 is 'release':
            sha1 = ''
        commit_hash_header = (
            '# DO NOT EDIT!  '
            'This file was automatically generated by setup.py of polytope')
        with open('polytope/commit_hash.txt', 'w') as f:
            f.write(commit_hash_header + '\n')
            f.write(sha1 + '\n')
    # Import polytope/version.py without importing polytope
    version = imp.load_module('version',
                              *imp.find_module('version', ['polytope']))
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
        requires=['numpy', 'scipy', 'networkx', 'cvxopt'],
        install_requires=[
            'numpy >= 1.6',
            'scipy',
            'networkx >= 1.6',
            'cvxopt == 1.1.7'],
        tests_require=[
            'nose',
            'matplotlib'],
        packages=[
            'polytope'],
        package_dir=dict(polytope='polytope'),
        package_data=dict(polytope=['commit_hash.txt']))
