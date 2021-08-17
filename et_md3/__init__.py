# -*- coding: utf-8 -*-

"""
Package et_md3
=======================================

Top-level package for et_md3.
"""

__version__ = "0.1.0"

import et_md3.verletlist.vlbuilders.grid

try:
    import et_md3.verletlist.verletlist_cpp
except ModuleNotFoundError as e:
    # Try to build this binary extension:
    from pathlib import Path
    import click
    from et_micc2.project import auto_build_binary_extension
    msg = auto_build_binary_extension(Path(__file__).parent, 'verletlist/cpp')
    if not msg:
        import et_md3.verletlist.verletlist_cpp
    else:
        click.secho(msg, fg='bright_red')

try:
    import et_md3.atoms.atoms_cpp
except ModuleNotFoundError as e:
    # Try to build this binary extension:
    from pathlib import Path
    import click
    from et_micc2.project import auto_build_binary_extension
    msg = auto_build_binary_extension(Path(__file__).parent, 'atoms/cpp')
    if not msg:
        import et_md3.atoms.atoms_cpp
    else:
        click.secho(msg, fg='bright_red')

import et_md3.time

import et_md3.verletlist.vlbuilders

import et_md3.potentials

import et_md3.verletlist

import et_md3.atoms


def hello(who='world'):
    """'Hello world' method.

    :param str who: whom to say hello to
    :returns: a string
    """
    result = "Hello " + who
    return result

# Your code here...