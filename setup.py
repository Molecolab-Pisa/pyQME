from setuptools import setup, find_packages

setup(
    name="pyQME",
    use_scm_version=True,           # <-- legge la versione dai tag Git
    setup_requires=['setuptools_scm'],
    url="https://molimen1.dcci.unipi.it/p.saraceno/pyQME",
    author="Piermarco Saraceno, Lorenzo Cupellini",
    author_email="piermarco.saraceno@phd.unipi.it",
    description=open("README.md").read(),
    packages=find_packages(),
    python_requires='>3.5.0',
    install_requires=["numpy", "scipy", "opt_einsum","tqdm","psutil","pathlib"],
)
