#!/bin/bash

. ~/anaconda3/bin/activate
#pip install coveralls
coverage run --source=phuzzy setup.py test
coveralls
