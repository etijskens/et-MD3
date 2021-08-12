# -*- coding: utf-8 -*-

"""
Package et_md3
=======================================

Top-level package for et_md3.
"""

__version__ = "0.1.0"

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