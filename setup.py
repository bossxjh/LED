from setuptools import find_packages, setup


setup(
    name="dataeval-led",
    version="0.1.0",
    description="LED metric and LIBERO data filtering utilities for robot datasets.",
    packages=find_packages(include=["dataeval", "dataeval.*", "scripts"]),
    python_requires=">=3.9",
)
