#!/usr/bin/env python
import sys
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='This is an expensive cli solver.')

parser.add_argument('-x', action="store", dest="x", type=float)
parser.add_argument('-i', '--input', action="store", dest="input")

args = parser.parse_args()

print(args)

x = float(args.x)
y = x**2
print(y)

# df = pd.read_csv(args.input)

with open("expensive_cli_expression.res", "w") as fhr:
    fhr.write("{}".format(y))
