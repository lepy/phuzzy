#!/usr/bin/env python
import os
import setuptools
from distutils.core import setup

def extract_version():
    init_py = os.path.join(os.path.dirname(__file__), "phuzzy", "__init__.py")
    version = "0.0.0"
    with open(init_py) as init:
        for line in init:
            if line.startswith("__version__"):
                version = line.split("=")[-1].strip().replace('"', '')
                print("version", version)
                return {"__version__": version}
        if version == "0.0.0":
            raise RuntimeError("Missing line starting with '__version__ =' in %s" % (init_py,))

DOWNLOAD_URL = 'https://github.com/lepy/phuzzy'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering']

with open(os.path.join('usage.rst')) as f:
    long_description = ''.join(f)

setup(name="phuzzy",
      description=("fuzzy data"),
      version=extract_version()["__version__"],
      author="Lepy",
      author_email="lepy@mailbox.org",
      url="https://github.com/lepy/phuzzy",
      packages=setuptools.find_packages(exclude=["tests"]),
      tests_require=[
          'pytest',
      ],
      long_description=long_description,
      classifiers=CLASSIFIERS,

      install_requires=['numpy', 'pandas', 'scipy'],

      )
