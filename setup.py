import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "wrfda_urb",
    version = "0.0.1",
    author = "Ronald van Haren",
    author_email = "r.vanharen@esciencecenter.nl",
    description = ("Implementation of data simulation for urban areas in WRF"),
    license = "Apache 2.0",
    keywords = "WRF, WRFDA, data assimilation",
    url = "https://github.com/ERA-URBAN/wrfda_urb",
    packages=['wrfda_urb'],
    scripts=['wrfda_urb/scripts/wrfda_urb'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved ::Apache Software License",
    ],
)

