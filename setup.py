import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='animius',
    version='1.0.0a1',
    author="gundamMC",
    author_email="gundamMC@gundamMC.com",
    description="A deep-learning virtual assistant engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://animius.org",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': ['animius = animius.Console:Console.start']
    },
    python_requires='>=3',
    install_requires=['numpy',
                      'pysubs2',
                      'pydub',
                      'scipy',
                      'speechpy',
                      'psutil',
                      'pynvml',
                      'librosa'],
    extras_require={
        "tf": ["tensorflow==1.12.0"],
        "tf_gpu": ["tensorflow-gpu==1.12.0"],
    },  # tf 2.0 not supported for now
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='animius ai virtual assistant deep learning'
)
