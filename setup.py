# !/usr/bin/env python

import os
import re
from setuptools import setup
try:  # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # pip <= 9.0.3
    from pip.req import parse_requirements


def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=False)
reqs = [str(ir.req) for ir in reqs]
reqs_dev = parse_requirements(resource('requirements-dev.txt'), session=False)
reqs_dev = [str(ir.req) for ir in reqs_dev]


with open(resource('sodinn', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)

with open(resource('README.md')) as readme_file:
    README = readme_file.read()

setup(
    name='sodinn',
    packages=['sodinn'],
    version=VERSION,
    description='Supervised detection of exoplanets',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Carlos Alberto Gomez Gonzalez',
    license='MIT',
    author_email='carlosgg33@gmail.com',
    url='https://github.com/carlgogo/sodinn',
    keywords=['deep-learning', 'exoplanets', 'detection', 'hci', 'package'],
    install_requires=reqs,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Astronomy'],
)
