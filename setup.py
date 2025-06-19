from setuptools import setup, find_packages

setup(
    name='pdf_chunker',                # Package name
    version='0.1.0',                   # Version following semantic versioning
    author='Aditya Kulkarni',
    author_email='aditya kulkarni',
    description='A brief description of your package',
    url='https://github.com/adityavkulkarni/pdf_chunker',  # Optional
    license='MIT',                     # Choose your license
    packages=find_packages(),          # Automatically find packages and subpackages
    install_requires=[                 # List dependencies here
        # 'numpy',
        # 'requests',
    ],
)
