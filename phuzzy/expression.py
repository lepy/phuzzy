# -*- coding: utf-8 -*-

import os
import subprocess
import numexpr

class ExpressionBase(object):
    """Approximate an expression of fuzzy numbers

    """

    def __init__(self, **kwargs):
        """Expression(kwargs)"""
        self.name = kwargs.get("name")

    def __call__(self, *args, **kwargs):
        """"""
        raise NotImplementedError()

    def __str__(self):
        return "({})".format(self.name)

    __repr__ = __str__


class Expression(ExpressionBase):
    """Approximate an expression of fuzzy numbers

    """

    def __init__(self, **kwargs):
        """Expression(kwargs)"""
        ExpressionBase.__init__(self, **kwargs)
        self.function = kwargs.get("function")

    def __call__(self, *args, **kwargs):
        """"""
        if self.function is None:
            raise Exception("function not defined")
        else:
            return self.function(*args, **kwargs)


class StrExpression(ExpressionBase):
    """Approximate an expression of fuzzy numbers

    """

    def __init__(self, **kwargs):
        """Expression(kwargs)"""
        ExpressionBase.__init__(self, **kwargs)
        self.function = kwargs.get("function")

    def __call__(self, *args, **kwargs):
        """"""
        print(kwargs)
        if self.function is None:
            raise Exception("function not defined")
        else:
            return numexpr.evaluate(self.function, local_dict=kwargs)


class CliExpression(ExpressionBase):
    """Approximate an expression of fuzzy numbers

    """

    def __init__(self, **kwargs):
        """Expression(kwargs)"""
        ExpressionBase.__init__(self, **kwargs)
        self.cmd = kwargs.get("cmd")
        self.workpath = kwargs.get("workpath") or os.path.dirname(__file__)

    def __call__(self, *args, **kwargs):
        """"""
        print(os.path.abspath(self.cmd))
        print(self.cmd)
        cmd = [self.cmd] + [str(x) for x in args]
        print(cmd)
        subprocess.call(cmd, shell=False, cwd=self.workpath)


if __name__ == '__main__':

    def f(x):
        return x ** 2


    x2 = Expression(name="x2", function=f)
    print(x2)
    print("x2(2)", x2(2))

    x2s = StrExpression(name="x2s", function="x**2")
    print(x2s)
    print("x2s(2)", x2s(x=2))


    x2c = CliExpression(name="x2c", cmd="../tests/f2.py")
    x2c(2)


    class F2(CliExpression):
        def __init__(self, **kwargs):
            CliExpression.__init__(self, **kwargs)

        def __call__(self, *args, **kwargs):
            """"""
            try:
                x = args[0]
                cmd = [self.cmd, "-x", "{}".format(x)]
                print(cmd)
                subprocess.call(cmd, shell=False, cwd=self.workpath)
                print(os.path.exists(os.path.join(self.workpath, "expensive_cli_expression.res")))

                y = float(open(os.path.join(self.workpath, "expensive_cli_expression.res")).read())
                return y
            except Exception as exp:
                print(exp)


    f2bin = os.path.abspath("../tests/expensive_cli_expression.py")
    f2 = F2(name="f2", cmd=f2bin, workpath="/tmp")
    y = f2(9)
    print("f2(2)", y)
