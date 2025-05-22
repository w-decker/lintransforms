from setuptools import setup, find_packages

setup(
    name="lintransforms",
    version="0.0.1",
    author="Will Decker",
    author_email="will.decker@gatech.edu",
    description="Apply various linear transformations on $n$ -dimensional matrices and solve for their transformation matrix.",
    url="https://github.com/w-decker/lintransforms",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ]

)