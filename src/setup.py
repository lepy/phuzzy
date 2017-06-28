# coding: utf-8

import os
import setuptools

def extract_version():
    init_py = os.path.join(os.path.dirname(__file__), "phuzzy", "__init__.py")
    with open(init_py) as init:
        for line in init:
            if line.startswith("__version__"):
                d = {}
                exec(line, d)
                return d["__version__"]
        else:
            raise RuntimeError("Missing line starting with '__version__ =' in %s" % (init_py,))


setup_params = dict(
    name="phuzzy",
    description = ("fuzzy data"),
    version=extract_version(),
    author="Lepy",
    author_email="lepy@mailbox.org",
    url="https://github.com/lepy/phuzzy",
    license = "MIT",
    keywords = "data, phuzzy",

    packages=setuptools.find_packages(exclude=["tests"]),
    zip_safe=False,

    install_requires=['numpy', 'pandas']

)

if __name__ == "__main__":
    setuptools.setup(**setup_params)
