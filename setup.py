#!/usr/bin/env python

# for testing on https://testpypi.python.org use:
#
# pip install -i https://testpypi.python.org/pypi polytope
#             --extra-index-url https://pypi.python.org/pypi
#
# so that dependencies can be found and installed,
# when testing in a clean virtualenv.

import os
from setuptools import setup

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
# installing the tulip package; checking occurs by default if
# "install" is given, unless both "install" and "nocheck" are given
# (but typical users do not need "nocheck").

# You *must* have these to run TuLiP.  Each item in other_depends must
# be treated specially; thus other_depends is a dictionary with
#
#   keys   : names of dependency;

#   values : list of callable and string, which is printed on failure
#           (i.e. package not found); we interpret the return value
#           True to be success, and False failure.
other_depends = {}

glpk_msg = 'GLPK seems to be missing\n' +\
    'and thus apparently not used by your installation of CVXOPT.\n' +\
    'If you\'re interested, see http://www.gnu.org/s/glpk/'
mpl_msg = 'matplotlib not found.\n' +\
    'For many graphics drawing features, you must install\n' +\
    'matplotlib (http://matplotlib.org/).'

# These are nice to have but not necessary. Each item is of the form
#
#   keys   : name of optional package;
#   values : list of callable and two strings, first string printed on
#           success, second printed on failure (i.e. package not
#           found); we interpret the return value True to be success,
#           and False failure.
optionals = {'glpk' : [check_glpk, 'GLPK found.', glpk_msg],
             'matplotlib' : [check_mpl, 'matplotlib found.', mpl_msg]}

import sys
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
                print(dep_val[1] )
                raise Exception('Failed dependency: '+dep_key)

    # Optional stuff
    for (opt_key, opt_val) in optionals.items():
        print('Probing for optional '+opt_key+'...')
        if opt_val[0]():
            print("\t"+opt_val[1] )
        else:
            print("\t"+opt_val[2] )

# load long description from README.rst
readme_file = 'README.rst'
if os.path.exists(readme_file):
    long_description = open(readme_file).read()
else:
    print('Could not find readme file to extract long_description.')
    long_description = ''

if perform_setup:
    # KEEP THIS NUMBER IN SYNC WITH THAT IN polytope/__init__.py
    polytope_version = '0.1.0'
    setup(
        name = 'polytope',
        version = polytope_version,
        description = 'Polytope Toolbox',
        long_description = long_description,
        author = 'Caltech Control and Dynamical Systems',
        author_email = 'polytope@tulip-control.org',
        url = 'http://tulip-control.org',
        license = 'BSD',
        requires = ['numpy', 'scipy', 'networkx', 'cvxopt'],
        install_requires = [
            'numpy >= 1.6', 'scipy', 'networkx >= 1.6', 'cvxopt'
        ],
        packages = [
            'polytope',
        ],
        package_dir = {'polytope' : 'polytope'},
        package_data={},
    )
