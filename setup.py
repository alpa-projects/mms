from setuptools import setup, find_packages

install_requires = [
    "numba",
    "scipy",
    "pulp",
    "matplotlib",
]

setup(name="alpa_serve",
      install_requires=install_require_list,
      packages=find_packages(exclude=["simulator"]))
