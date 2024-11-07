from setuptools import setup, find_packages

setup(
    name="pyQME",
    version="0.0.1",
    url="https://molimen1.dcci.unipi.it/p.saraceno/pyQME",
    author="Piermarco Saraceno, Lorenzo Cupellini",
    author_email="piermarco.saraceno@phd.unipi.it",
    description=open("README.md").read(),
    packages=find_packages(),
    install_requires=["numpy", "scipy", "opt_einsum","tqdm","psutil","warnings"],
)
