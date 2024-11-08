from setuptools import setup, find_packages

setup(
    name='Cobra',
    version='2.0',
    description='Bayesian Pulsar searching',
    author='Lindley Lentati, Prajwal Padmanabh',
    author_email='lindleylentati@gmail.com, prajwal3108@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pymultinest',
        'matplotlib',
        'corner',
        'scipy',
        'cupy',
        'libstempo'
    ],
    python_requires='>=3.6',
)
