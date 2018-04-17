#!/bin/bash

. ~/anaconda3/bin/activate
pip install collective.checkdocs
python setup.py checkdocs
