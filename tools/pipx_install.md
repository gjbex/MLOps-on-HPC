# Installing DVC on HPC systems

The most convenient way to install DVC on HPC systems is by using `pipx`, which
allows you to install and run Python applications in isolated environments.
This method avoids conflicts with system packages and other Python
applications.

If pipx can not be installed on your HPC system, a virtual environment can be used
as an alternative.


## Prerequisites

On most HPC systems, Python and pip will be available as part of the system
image.  However, it is likely that it is an older version. In that case, you can
install DVC, but  it will be an older version as well.

Load a Python module if available:

```bash
$ module load python
```

Create a virtual environment to install pipx.  (This step is not required if
pipx is already available on your system, or it can be installed usring `python
-m pip install --user pipx`.)

```bash
$ python3 -m venv ~/pipx_venv --system-site-packages
$ source ~/pipx_venv/bin/activate
```

Install pipx using pip and initialize::

```bash
$ python -m pip install pipx
$ python -m pipx ensurepath
```

Install DVC using pipx:

```bash
$ pipx install dvc[ssh]
```

Now you can deactivate the virtual environment and unload the Python module if
needed:

```bash
$ deactivate
$ module unload python
```

Now DVC is installed and ready to use on your HPC system.


## Troubleshooting

Although applications installed with pipx are isolated from the system Python
packages, things can go south if `PYTHONPATH` is set.  If that is the case, you
can run the applications nevertheless, e.g.,

```bash
$ PYTHONPATH= dvc --version
```

Alternatively, you can unset the `PYTHONPATH` variable in the following way:

```bash
$ env -u PYTHONPATH dvc --version
```

More conveniently, you can define an alias in your shell configuration file:

```bash
alias dvc='env -u PYTHONPATH dvc'
```
