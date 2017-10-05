Requirements files for [`pip`](https://github.com/pypa/pip). Dependencies are
pinned to ensure reproducibility, unlike `install_requires` in `setup.py`,
which attempts to leave the user's environment unconstrained.

The requirements are organized by level of necessity:

1. `default.txt`: necessary
2. `extras.txt`: optional
3. `test.txt`: some tests fail without these

The file `../requirements.txt` omits extras.
