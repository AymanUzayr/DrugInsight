from setuptools import setup, find_packages

setup(
    name='druginsight',
    version='0.1.0',
    description='Explainable drug-drug interaction prediction framework',
    author='DrugInsight',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'rdkit>=2023.3.1',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'fastapi>=0.100.0',
        'uvicorn>=0.23.0',
        'streamlit>=1.28.0',
        'pydantic>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'druginsight=drug_insight.cli:main',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
    ],
)
