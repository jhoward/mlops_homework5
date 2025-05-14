from setuptools import setup, find_packages

setup(
    name="mlops_homework5",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "sentence-transformers",
        "scikit-learn",
        "pandas",
        "pytest",
    ],
) 