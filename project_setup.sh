#!/bin/bash

# Install the project in editable mode
pip install -e .

# Create and install the IPython kernel for the project
#python -m ipykernel install --user --name=mlx_stuff --display-name "Entropix"
python -m ipykernel install --sys-prefix --name=mlx_stuff --display-name "Entropix"


echo "Jupyter kernel 'Entropix' has been installed."