# -*- coding: utf-8 -*-
"""Tests for phuzzy.expression.

Not to be confused with test_expresssion.py, which covers the unrelated
phuzzy.approx.doe.Expression.
"""

import os
import stat
import sys

import numpy as np
import pytest

from phuzzy.expression import CliExpression, Expression, ExpressionBase, StrExpression


def _f(x):
    return x ** 2


def test_expression_base_str():
    e = ExpressionBase(name="base")
    assert e.name == "base"
    assert str(e) == "(base)"
    assert repr(e) == "(base)"


def test_expression_base_name_defaults_to_none():
    assert ExpressionBase().name is None


def test_expression_base_is_not_callable():
    with pytest.raises(NotImplementedError):
        ExpressionBase(name="base")()


def test_expression_calls_its_function():
    x2 = Expression(name="x2", function=_f)
    assert str(x2) == "(x2)"
    assert x2(2) == 4
    assert x2(x=3) == 9


def test_expression_without_function_raises():
    with pytest.raises(Exception, match="function not defined"):
        Expression(name="x2")()


def test_str_expression_evaluates_via_numexpr():
    x2s = StrExpression(name="x2s", function="x**2")
    assert str(x2s) == "(x2s)"
    assert x2s(x=2) == pytest.approx(4.0)


def test_str_expression_evaluates_arrays():
    expr = StrExpression(name="sum", function="x + y")
    result = expr(x=np.array([1.0, 2.0]), y=np.array([10.0, 20.0]))
    assert result == pytest.approx([11.0, 22.0])


def test_str_expression_without_function_raises():
    with pytest.raises(Exception, match="function not defined"):
        StrExpression(name="x2s")()


def _write_script(path, body):
    path.write_text("#!{}\n{}".format(sys.executable, body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_cli_expression_runs_command(tmp_path):
    script = _write_script(tmp_path / "square.py", (
        "import sys\n"
        "x = float(sys.argv[1])\n"
        "open('result.res', 'w').write(str(x ** 2))\n"
    ))

    cli = CliExpression(name="x2c", cmd=str(script), workpath=str(tmp_path))
    assert cli(3) is None

    assert (tmp_path / "result.res").read_text() == "9.0"


def test_cli_expression_runs_in_workpath(tmp_path):
    """The command runs with cwd=workpath, so relative writes land there."""
    workpath = tmp_path / "work"
    workpath.mkdir()
    script = _write_script(tmp_path / "cwd.py", (
        "import os\n"
        "open('cwd.res', 'w').write(os.getcwd())\n"
    ))

    CliExpression(name="cwd", cmd=str(script), workpath=str(workpath))(1)

    assert (workpath / "cwd.res").read_text() == str(workpath)


def test_cli_expression_defaults_workpath_to_module_dir():
    cli = CliExpression(name="x2c", cmd="/bin/true")
    assert cli.workpath == os.path.dirname(sys.modules["phuzzy.expression"].__file__)
