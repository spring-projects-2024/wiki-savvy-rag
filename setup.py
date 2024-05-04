from setuptools import setup, find_packages

setup(
    name='backend',                       # Choose a unique name for your package
    version='0.1.0',                        # Start with a small version number
    packages=find_packages(where='.'),      # This automatically finds packages in the provided directory
    package_dir={'': '.'},              # Tells setuptools that packages are under the provided directory
)
