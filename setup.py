# coding: utf-8

import os
import setuptools

def extract_version():
    init_py = os.path.join(os.path.dirname(__file__), "phuzzy", "__init__.py")
    version = "0.0.0"
    with open(init_py) as init:
        for line in init:
            if line.startswith("__version__"):
                version = line.split("=")[-1].strip().replace('"', '')
                print("version", version)
                return {"__version__":version}
        if version == "0.0.0":
            raise RuntimeError("Missing line starting with '__version__ =' in %s" % (init_py,))


setup_params = dict(
    name="phuzzy",
    description = ("fuzzy data"),
    version = extract_version()["__version__"],
    author = "Lepy",
    author_email="lepy@mailbox.org",
    url="https://github.com/lepy/phuzzy",
    license = "MIT",
    keywords = "data, phuzzy",

    packages=setuptools.find_packages(exclude=["tests"]),
    zip_safe=False,

    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],

    install_requires=['numpy', 'pandas', 'scipy'],
)


if __name__ == "__main__":
    setuptools.setup(**setup_params)
