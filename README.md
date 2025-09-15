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

## Citing
If you use `ising2D` in your research, please cite:

Qianyang Chen, Nihat Ay and Mikhail Prokopenko, *Generalizing thermodynamic efficiency of interactions: inferential, information-geometric and computational perspectives*, arXiv:2509.10102 [nlin.AO] (2025).  
Available at: [https://arxiv.org/abs/2509.10102](https://arxiv.org/abs/2509.10102)

### BibTeX
```bibtex
@article{chen2025generalize,
  author  = {Chen, Qianyang and Ay, Nihat and Prokopenko, Mikhail},
  title   = {Generalizing thermodynamic efficiency of interactions: inferential, information-geometric and computational perspectives},
  journal = {arXiv preprint arXiv:2509.10102},
  year    = {2025},
  eprint  = {2509.10102},
  archivePrefix = {arXiv},
  primaryClass = {nlin.AO}
}
