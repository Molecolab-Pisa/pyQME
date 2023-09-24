from setuptools import setup

setup(
    name="redfield_package",
    version="0.0.1",
    url="https://molimen1.dcci.unipi.it/p.saraceno/redfield-package",
    author="Piermarco Saraceno, Lorenzo Cupellini",
    author_email="piermarco.saraceno@phd.unipi.it",
    description=open("README.md").read(),
    packages=["redfield_package"],
    install_requires=["numpy", "scipy", "opt_einsum"],
)
