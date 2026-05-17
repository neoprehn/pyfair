from setuptools import setup

setup(
    name="pyfair",
    version="1.0.0",
    description="Open FAIR Monte Carlo creator",
    long_description="""
        Factor Analysis of Information Risk (Open FAIR) model in Python.

        This package endeavors to create a simple API for automating the
        creation of FAIR Monte Carlo risk simulations.

        This is based on the terms found in:

        1. Open FAIR™ RISK TAXONOMY (O-RT); and,
        2. Open FAIR™ RISK ANALYSIS (O-RA).

        "Open FAIR" is a trademark of the Open Group.

    """,
    author="Mirko Prehn",
    author_email="mirko.prehn@web.de",
    packages=[
        "pyfair",
        "pyfair.model",
        "pyfair.report",
        "pyfair.utility",
    ],
    license="MIT",
    url="https://github.com/Hive-Systems/pyfair",
    keywords=["FAIR", "risk"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "pandas>=0.24.1",
        "numpy>=1.16.1",
        "scipy>=1.2.1",
        "matplotlib>=3.0.2",
        "xlrd>=1.2.0",
    ],
    package_data={"pyfair": ["static/*"]},
)
