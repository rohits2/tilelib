import os
import sys

from setuptools import setup, find_packages


open_kwds = {}
if sys.version_info > (3,):
    open_kwds["encoding"] = "utf-8"

#with open("README.rst", **open_kwds) as f:
    #readme = f.read()

setup(
    name="tilelib",
    version="1.0.0",
    description="Tile-based road and image utilities",
    #long_description=readme,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    keywords="mapping, web mercator, tiles, tilestacks, openstreetmap",
    author="Rohit Singh",
    author_email="singhrohit2@hotmail.com",
    #url="https://github.com/mapbox/mercantile",
    #license="BSD",
    packages=find_packages(exclude=["ez_setup", "examples", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=["mercantile","pillow","geopandas", "pandas", "osmnx", "numpy", "shapely"],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coveralls", "pytest-cov", "pydocstyle"],
    },
)
