"""
================================================================================
pyDOE: Design of Experiments for Python
================================================================================

This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez

    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by
Abraham Lee.

License
=======

This package is provided under two licenses:

1. The *BSD License* (3-clause)
2. Any other that the author approves (just ask!)

"""

# from __future__ import absolute_import

__author__ = 'Abraham Lee'
__version__ = '0.3.8'

from phuzzy.contrib.pydoe.doe_box_behnken import *
from phuzzy.contrib.pydoe.doe_composite import *
from phuzzy.contrib.pydoe.doe_factorial import *
from phuzzy.contrib.pydoe.doe_lhs import *
from phuzzy.contrib.pydoe.doe_fold import *
from phuzzy.contrib.pydoe.doe_plackett_burman import *
# from phuzzy.contrib.pydoe.var_regression_matrix import *

