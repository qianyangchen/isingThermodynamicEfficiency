# isingThermodynamicEfficiency
2D Ising model simulation for computing thermodynamic efficiency (inferential form, info-geometric form and computation form).

## Installation
1. Clone the repository
```
git clone https://github.com/qianyangchen/isingThermodynamicEfficiency.git
cd isingThermodynamicEfficiency
```

2. Install the core package (simulation only)
This will install ising2D and the minimal dependencies needed to run simulations:
```
pip install -e .
```

You can now use the package in Python:
```
from ising2D import core
```

3. Install with extras
To use the example analysis notebook:
```
pip install -e .[analysis]
```
