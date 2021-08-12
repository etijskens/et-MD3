#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for sub-module et_md3.verletlist.vlbuilders."""

import pytest
import et_md3.verletlist.vlbuilders


def test_greet():
    expected = "Hello John!"
    greeting = et_md3.vlbuilders.greet("John")
    assert greeting==expected


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_greet

    the_test_you_want_to_debug()
    print("-*# finished #*-")
# ==============================================================================
