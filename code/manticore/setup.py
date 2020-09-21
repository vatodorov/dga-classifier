########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from setuptools import setup, find_packages

setup(
    name='mc-dga-classifier',
    version='0.0.3',
    description='API server that serves the DGA classifier algorithm',
    author='Awesome Developer',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'License :: Proprietary License'
    ],
    entry_points={
        'console_scripts': [
            'mc-dga-classifier=mc_dga_classifier.app:run'
        ]
    },
    zip_safe=False,
    install_requires=[
        'Flask==1.1.2',
        'PyYAML>=5.3.1'
    ],
    python_requires='>=3.6'
)
