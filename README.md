This is the official code base for Era Splitting, the subject of a recent research paper. Using this repository you can install and run the HistGradientBoostingRegressor with the new era splitting or directional era splitting criteria implemented via simple arguments.


# Installation

## Clone the Repo

```
git clone https://github.com/jefferythewind/scikit-learn-erasplit.git
```

## Install via Pip

```
cd scikit-learn/
pip install .
```

# Example Implementation w/ Numerai Data

```python

from pathlib import Path
from numerapi import NumerAPI #pip install numerapi
import json

"""Era Split Model"""
from sklearn.ensemble import EraHistGradientBoostingRegressor

napi = NumerAPI()
Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/train.parquet")
napi.download_dataset("v4/features.json")

with open("v4/features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]['small']
TARGET_COL="target_cyrus_v4_20"

training_data = pd.read_parquet('v4/train.parquet')
training_data['era'] = training_data['era'].astype('int')

model = EraHistGradientBoostingRegressor( 
    early_stopping=False, 
    boltzmann_alpha=0, 
    max_iter=5000, 
    max_depth=5, 
    learning_rate=.01, 
    colsample_bytree=.1, 
    max_leaf_nodes=32, 
    gamma=0, #for era splitting
    blama=0  #for directional era splitting
)
model.fit(training_data[ features ], training_data[ TARGET_COL ], training_data['era'].values)
```

## Explanation of Parameters
### Boltzmann Alpha
The Boltzmann alpha parameter varies from -infinity to +infinity. A value of zero recovers the mean, -infinity recovers the minumum and +infinity recovers the maximum. This smooth min/max function is applied to the era-wise impurity scores when evaluating a data split. For all the experiments in the paper, we use the default setting of alpha=0. 

Read more: https://en.wikipedia.org/wiki/Smooth_maximum
### Gamma
The Gamma parameter varies over the interval [0,1]. This tells us how much of the original splitting criterion to mix in with the era splitting criterion. The default setting of zero means we want to use 100% the era splitting criterion. If we give a value of 0.5, that means our split criterion will be 50% the era splitting criterion and 50% the original splitting criterion. For our state of the art Numerai model, we used a value of 0.6 here.

### Blama
The Blama parameter is similar to the Gamma parameter, but here we measure how much of the directional splitting criterion we want to mix in to our model. If we want pure directional era splitting, which is common in the research paper, we set Blama = 1.

One can also mix the era splitting and directional era splitting criteria via the following formula
```python
gain = ( 1 - blama - gamma ) * era_split_gain + blama * directional_era_split_gain + gamma * original_gain
```
Example, if I wanted 50% era splitting and 50% direcitonal era splitting, I would set Blama = 0.5. If we want 50% original splitting criterion and 50% directional era splitting criterion, then we would set Blama = 0.5 and Gamma = 0.5.

# Complete Code Notebook Examples Available here:

# Citations:

This code was forked from the official scikit-learn repository and is currently a stand-alone version. All community help is welcome for getting these ideas part of the official scikit learn code base or even better, incorporated in the LightGBM code base.
