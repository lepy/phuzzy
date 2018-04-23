# -*- coding: utf-8 -*-

__title__ = "phuzzy"
__author__ = "lepy"
__email__ = "lepy@mailbox.org"
__description__ = """Fuzzy stuff"""
__long_description__ = """
fuzzy number tools
"""
__url__ = 'https://github.com/lepy/phuzzy'
__copyright__ = "Copyright (C) 2018-"
__version__ = "0.6.4"
__status__ = "3 - Alpha"
__credits__ = [""]
__license__ = """MIT"""

import logging
logger = logging.getLogger("phuzzy")

from phuzzy.shapes import FuzzyNumber, Trapezoid, Triangle, Uniform
from phuzzy.shapes.superellipse import Superellipse
from phuzzy.shapes.truncnorm import TruncGenNorm, TruncNorm

