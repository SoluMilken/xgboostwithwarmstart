#!/usr/bin/env python
from setuptools import setup

install_requires = [
    'xgboost',
    'numpy',
]

setup(
    name='xgboostwithwarmstart',
    version="0.1.0",
    description='XGBoost can warm start !!!!',
    long_description=("See `github <'https://github.com/SoluMilken/xgboostwithwarmstart'>`_ "
                      "for more information."),
    author='ianlini and solumilken',
    url='https://github.com/SoluMilken/xgboostwithwarmstart',
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 1 - Planning',

        'Intended Audience :: Developers',
        'Topic :: Utilities',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=[
        'xgboostwithwarmstart',
    ],
    package_dir={
        'xgboostwithwarmstart': 'xgboostwithwarmstart',
    },
)
