import codecs
import os
from distutils.core import setup

from setuptools import find_packages

setup_path = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(setup_path, 'README.md'), encoding='utf-8-sig') as f:
    README = f.read()

setup(name='infodeslib',
      version='0.0.5',
      url='https://github.com/fukashi-hatake/infodeslib',
      maintainer='Firuz Juraev',
      maintainer_email='f.i.juraev@gmail.com',
      description='Implementation of Dynamic Ensemble Selection methods for Late Fusion',
      long_description=README,
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
