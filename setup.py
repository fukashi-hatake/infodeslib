from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Dynamic Ensembple Selection Library for Classification with Late fusion setting.'
LONG_DESCRIPTION = 'Dynamic Ensembple Selection Library for Classification with Late fusion setting.' 

# Setting up
setup(
    name="infodeslib",
    version=VERSION,
    author="Firuz Juraev",
    author_email="<f.i.juraev@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['sklearn', 'matplotlib', 'numpy'],
    keywords=['python', 'des', 'classification', 'latefusion'],
    classifiers=[
        "Development Status :: First release",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)