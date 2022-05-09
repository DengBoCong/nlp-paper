#! -*- coding: utf-8 -*-

import codecs
import pathlib
from setuptools import setup
from setuptools import find_packages

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="paper-kits",
    version="1.0.1",
    description="Deep learning paper kits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/DengBoCong/nlp-paper",
    author="DengBoCong",
    author_email="bocongdeng@gmail.com",
    install_requires=["jieba==0.42.1", "PyQt6==6.3.0"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="paper, nlp, deep learning, kits",
    project_urls={
        "Bug Reports": "https://github.com/DengBoCong/nlp-paper/issues",
        "Funding": "https://pypi.org/project/paper-kits/",
        "Source": "https://github.com/DengBoCong/nlp-paper",
    },
)
