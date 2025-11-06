from setuptools import setup, find_packages

setup(
    name="lll-cognitive-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    description="A cognitive core system for AI agents with memory and reasoning",
    python_requires=">=3.8",
)
