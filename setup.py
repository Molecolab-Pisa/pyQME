from setuptools import setup

setup(
    name="pyQME",
    version="0.0.1",
    url="https://molimen1.dcci.unipi.it/p.saraceno/pyQME",
    author="Piermarco Saraceno, Lorenzo Cupellini",
    author_email="piermarco.saraceno@phd.unipi.it",
    description=open("README.md").read(),
    packages=["pyQME"],
    install_requires=["numpy", "scipy", "opt_einsum"],
)
