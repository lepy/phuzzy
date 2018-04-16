#!/bin/bash
cd ~/projects/phuzzy
rm dist/*.tar.gz
python setup.py build sdist upload
twine upload dist/*

