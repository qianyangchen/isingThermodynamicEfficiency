from setuptools import setup, find_packages

setup(
    name="ising2D",                    
    version="0.1.0",
    description="2D Ising model simulation package",
    author="Qianyang Chen",
    license="MIT",                     
    packages=find_packages(where="src"),
    package_dir={"": "src"},          
    install_requires=[
        "numpy",
        "joblib",
        "numba",
        "scipy",
    ],
    extras_require={
        "analysis": ["ipympl", "matplotlib", "notebook", "jupyterlab"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

