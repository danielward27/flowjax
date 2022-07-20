import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="flowjax",
    version="3.0.0",
    url="https://github.com/danielward27/flowjax.git",
    license="MIT",
    author="Daniel Ward",
    author_email="danielward27@outlook.com",
    description="Normalizing flow implementations in jax.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "jax",
        "jaxlib>=0.3",
        "equinox",
        "tqdm",
        "optax",
        "numpy<=1.22.4"  # https://github.com/google/jax/issues/11241
        ],
    extras_require={
        'dev': ['pytest']
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
