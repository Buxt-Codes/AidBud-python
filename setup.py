from setuptools import setup, find_packages

with open("aidbud/requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="aidbud",
    version="0.1",
    packages=find_packages(include=["aidbud", "aidbud.*"]),
    install_requires=requirements,
)