# MIGC

This is the code of MIGC.

## Environment:
```
conda create -n MIGC_env python=3.8.12
conda activate MIGC_env
pip install pandas==1.2.3 
pip install anndata==0.8.0 
pip install scanpy==1.8.2
pip install numpy==1.21.6
pip install scipy==1.10.1 
pip install numba==0.57.0 
pip install matplotlib==3.3.4
pip install scvelo==0.2.4
pip install typing_extensions
pip install torch==1.10.1+cu113
```

## Reproduce:
Notebooks for reproducing the results on different datasets are provided in MIGC_demo_10x_mouse_brain.ipynb and MIGC_demo_gastrulation_erythroid.ipynb.
