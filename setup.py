import os
from setuptools import setup, find_packages

setup(
    name='updated_sphericalCNN',
    version="1.0.0",
    author="Jeremy Funited",
    url="https://github.com/funited/updated_sphericalCNN",
    
    # 1. Automatically finds your Python folders (requires an __init__.py in the folder)
    packages=find_packages(exclude=["build", "tests", "docs"]),
    
    # 2. CRITICAL: The libraries your code needs to run
    install_requires=[
        'torch>=1.10.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0'
        # Add any specific SHT libraries you use here (e.g., 's2fft')
    ],
    
    # 3. Ensures someone doesn't try to install this on an ancient Python version
    python_requires='>=3.7',
)
