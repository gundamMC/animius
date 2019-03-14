import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='animius',
     version='1.0',
     author="gundamMC",
     author_email="gundamMC@gundamMC.com",
     description="A deep-learning virtual assistant engine",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/gundamMC/Animius",
     packages=setuptools.find_packages(),
     python_requires='>=3',
     install_requires=['numpy', 'pysubs2', 'pydub', 'scipy', 'speechpy'],
     extras_require={
        "tf": ["tensorflow==1.12.0"],
        "tf_gpu": ["tensorflow-gpu==1.12.0"],
     }  # tf 2.0 not supported for now
)
