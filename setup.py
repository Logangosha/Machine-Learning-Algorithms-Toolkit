from setuptools import setup, find_packages

setup(
    name="ml_algorithms",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "matplotlib",
        "pandas"
    ],
    author="Logan Gauchat",
    description="Collection of ML algorithms in Python",
)