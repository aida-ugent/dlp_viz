# Code accompanying the paper "New Perspectives on the Evaluation of Link Prediction Algorithms for Dynamic Graphs"

`tgn_attn` contains a simple training script for the TGN-Attn model. 

The code used to generate the plots in the paper is available in the `notebooks` folder as jupyter notebook.

# Getting Started

## Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## Install the requirements
```bash
pip install -r requirements.txt
```



# Model Weights
In the examples plots provided, pre-trained models are used. 
I can provide the weights for the models upon request.
Else the scores and weights can be obtained using:
- The `tgn_attn` script for the score against selected negative samples experiment
- By adapting the code in [https://github.com/shenyangHuang/TGB/blob/main/examples/linkproppred/tgbl-wiki/tgn.py](https://github.com/shenyangHuang/TGB/blob/main/examples/linkproppred/tgbl-wiki/tgn.py)

For any questions, please contact [Raphaël Romero](raphael.romero@ugent.be).