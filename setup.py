from setuptools import setup, find_packages

setup(
    name='PyMergers',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'gwpy==3.0.8',
        'pycbc==2.6.0',
        'scipy==1.13.1',
        'tensorflow==2.12.0',
        'tqdm==4.66.5',
    ],
    python_requires='>=3.9',
    include_package_data=True,
    package_data={'PyMergers': ['models/*.tflite']},
    entry_points={
        'console_scripts': [
            'pymergers=PyMergers.main:main',  # Pointing to the main function
        ],
    },
    author='Wathela Alhassan',
    author_email='wathelahamed@gmail.com',
    description='Detect Binary Black Hole mergers from Einstein Telescope data using Deep Convolutional Neural Networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wathela/PyMerger', 
    license='MIT', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
    ],
)