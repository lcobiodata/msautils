from setuptools import setup, find_packages

setup(
    name='protlearn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'argparse',
        'numpy',
        'scipy',
        'matplotlib',
        'networkx',
        'biopython',
        'pyclustering',
        'scikit-learn',
        'pandas',
        'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'my_script = my_package.my_script:main'
        ]
    }
)
