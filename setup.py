from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyQME",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    url="https://molimen1.dcci.unipi.it/p.saraceno/pyQME",
    author="Piermarco Saraceno, Lorenzo Cupellini",
    author_email="piermarco.saraceno@phd.unipi.it",
    description="Package for Spectra Simulation and approximate Quantum Master Equations in exciton aggregates.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy<2.0",
        "scipy>=1.10,<2.0",
        "opt_einsum",
        "tqdm",
        "psutil",
    ],
)
