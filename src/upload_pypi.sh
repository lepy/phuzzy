#!/bin/bash
cd ~/projects/phuzzy/src
rm dist/*.tar.gz
python setup.py sdist upload
twine upload dist/*

