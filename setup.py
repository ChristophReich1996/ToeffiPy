from setuptools import setup

setup(
    name='ToeffiPy',
    version='1.0',
    description='ToeffiPy is a simple autograd/deep learning library based only on numpy.',
    author='Christoph Reich',
    author_email='ChristophReich@gmx.net',
    packages=['autograd', 'autograd.data', 'autograd.nn'],
    install_requires=['numpy>=1.16.2'],
    keywords=['Automatic differentiation', 'backpropagation', 'deep learning'
              'machine learning', 'optimization', 'neural networks',
              'Python', 'Numpy'],
    url='https://github.com/ChristophReich1996/ToeffiPy',
    license='MIT',
    classifiers=['Programming Language :: Python :: 3.7']
)