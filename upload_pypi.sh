#!/bin/bash
cd ~/projects/phuzzy
rm dist/*.tar.gz
python setup.py build sdist
twine upload dist/*

