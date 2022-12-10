from setuptools import setup, find_packages

install_requires = [
    "numba",
    "scipy",
    "pulp",
    "matplotlib",
    "starlette",
    "uvicorn",
    "fastapi",
]

setup(name="alpa_serve",
      install_requires=install_requires,
      packages=find_packages(exclude=["simulator"]))
