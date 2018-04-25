# -*- coding: utf-8 -*-

import phuzzy
import phuzzy.approx.doe

def eval(expression):
    pass

class FuzzyNumber(phuzzy.FuzzyNumber):
    """

    """
    def __init__(self, **kwargs):
        """Approximated FuzzyNumber with reduced operations during alpha-level optimization

        :param kwargs:
        """
        phuzzy.FuzzyNumber.__init__(self, **kwargs)
        self.samples = None # sampled input data sets [alpha, x, y, ...]
        self.df_res = None # dataframe [alpha, res]
        self.model = None # regression model (e.g. SVM -> prediction = model.predict(T))

if __name__ == '__main__':
    f = FuzzyNumber()
    print(f)
