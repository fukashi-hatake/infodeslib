import codecs
import os
from pathlib import Path
from distutils.core import setup

from setuptools import find_packages

THIS_DIRECTORY = Path(__file__).parent 

readme_path = THIS_DIRECTORY / ".." / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text() 

setup(name='infodeslib',
      version='0.0.6',
      url='https://github.com/fukashi-hatake/infodeslib',
      maintainer='Firuz Juraev',
      maintainer_email='f.i.juraev@gmail.com',
      description='Implementation of Dynamic Ensemble Selection methods with Late Fusion',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Firuz Juraev',
      author_email='f.i.juraev@gmail.com',
      license='MIT',

      install_requires=[
          'scikit-learn>=0.21.0',
          'numpy>=1.17.0',
          'scipy>=1.4.0',
      ],
      python_requires='>=3',      

      packages=find_packages())
