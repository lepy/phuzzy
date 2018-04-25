# -*- coding: utf-8 -*-

import phuzzy
import phuzzy.approx.doe


class FuzzyNumber(phuzzy.FuzzyNumber):
    """

    """

    def __init__(self, **kwargs):
        """Approximated FuzzyNumber with reduced operations during alpha-level optimization

        :param kwargs:
        """
        phuzzy.FuzzyNumber.__init__(self, **kwargs)
        self.samples = None
        self.df_res = None


if __name__ == '__main__':
    f = FuzzyNumber()
    print(f)
