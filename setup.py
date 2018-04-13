from setuptools import setup

setup(name='kineticmodel',
      version='0.1.0',
      description='PET kinetic models',
      url='http://github.com/bilgelm/kineticmodel',
      author='Murat Bilgel',
      author_email='murat.bilgel@nih.gov',
      license='MIT',
      packages=['kineticmodel'],
      install_requires=[
          'temporalimage',
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose','ddt'],
      extras_require={'nipype': ['nipype']})
