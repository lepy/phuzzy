# -*- coding: utf-8 -*-

__title__ = "phuzzy"
__author__ = "lepy"
__email__ = "lepy@tuta.io"
__description__ = """Fuzzy stuff"""
__long_description__ = """
fuzzy number tools
"""
__url__ = 'https://github.com/lepy/phuzzy'
__copyright__ = "Copyright (C) 2018-"
__version__ = "0.8.0"
__status__ = "3 - Alpha"
__credits__ = [""]
__license__ = """MIT"""

import logging
logger = logging.getLogger("phuzzy")

from phuzzy.shapes import FuzzyNumber, Trapezoid, Triangle, Uniform, Constant
from phuzzy.shapes.superellipse import Superellipse
from phuzzy.shapes.truncnorm import TruncGenNorm, TruncNorm

class Analysis(object):
    def __init__(self, **kwargs):
        """Analysis(kwargs)"""

        self.name = kwargs.get("name", "FuzzyAnalysis N.N.")
        self._designvars = []

        if "designvars" in kwargs:
            self.add_designvars(kwargs.get("designvars"))

    def __str__(self):
        return "(Analysis:'{o.name}', dv={d}".format(o=self,
                                                     d=self._designvars,
                                                     )

    __repr__ = __str__

    @property
    def designvars(self):
        """returns all design variables of doe

        :return: dict of designvars
        """
        return self._designvars

    def add_designvar(self, designvar):
        """add design variable to doe

        :param designvar: design variable
        :return: None
        """
        self._designvars.append(designvar)

    def add_designvars(self, designvars):
        """add design variables to doe

        :param designvars: list of design variables
        :return: None
        """
        self._designvars.extend(designvars)
