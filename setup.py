from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='land_use_prediction_AML_course',
    version='0.1.0',
    author='Tobias Liechti, Jan Göpel',
    author_email='tobiliechti@gmail.com, Jan.Goepel@wyssacademy.org',
    description='Final project for 2023 CAS in Advanced Machine Learning at the University of Bern. Predicts land use changes using Markov Chain transition probabilities. Analyzes observed changes to forecast future states.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Moonbear2457/land_use_prediction_AML_course',
    packages=find_packages(),
    install_requires=[
        'geopandas>=0.9.0',
        'rasterio>=1.2.0',
        'matplotlib>=3.5.0',
        'pandas>=1.4.0',
        'numpy>=1.19.5',
        'scipy>=1.6.0',
        'scikit-learn>=0.24.2',
        'statsmodels>=0.12.0',
        'tensorflow>=2.4.0',
        'seaborn>=0.11.1',
        'shapely>=1.8.0',
        'fiona>=1.8.21',
        'pyproj>=3.3.0',
        'packaging',
        'joblib>=1.2.0',
        'threadpoolctl>=3.1.0',
        'cython>=3.0.10'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    test_suite='tests',
)

