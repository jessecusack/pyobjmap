from setuptools import setup, find_packages

setup(name='pyobjmap',
      version='0.1.0',
      description='Optimal interpolation/objective mapping function for python',
#      long_description=open('README.rst').read(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Oceanography :: Data analysis',
      ],
      url='http://github.com/jessecusack/pyobjmap',
      author='Jesse Cusack',
      author_email='jmcusack@ucsd.edu',
#      license='MIT',
      packages=find_packages(include=["pyobjmap", "pyobjmap.*"], exclude=["data", "docs"]),
      install_requires=[
          'numpy', 'scipy',
      ],
      python_requires='>=3.6',
      zip_safe=False)
