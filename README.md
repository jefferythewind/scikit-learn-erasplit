This is the official code base for Era Splitting. Using this repository you can install and run the EraHistGradientBoostingRegressor with the new **era splitting**, **directional era splitting**, or original criterion implemented via simple arguments.

Era Splitting Paper: https://arxiv.org/abs/2309.14496

# Installation

## Clone the Repo

```
git clone --single-branch --branch era_splitting-tiebreaker https://github.com/jefferythewind/scikit-learn-erasplit.git
```

## Install via Pip

```
cd scikit-learn-erasplit/
pip install .
```

# Example Implementation w/ Numerai Data

```python

from pathlib import Path
from numerapi import NumerAPI #pip install numerapi
import json

"""Era Split Model"""
from erasplit.ensemble import EraHistGradientBoostingRegressor

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
    gamma=1, #for era splitting
    #blama=1,  #for directional era splitting
    #vanna=1,  #for original splitting criterion
)
model.fit(training_data[ features ], training_data[ TARGET_COL ], training_data['era'].values)
```

## Explanation of Parameters
### Boltzmann Alpha
The Boltzmann alpha parameter varies from -infinity to +infinity. A value of zero recovers the mean, -infinity recovers the minumum and +infinity recovers the maximum. This smooth min/max function is applied to the era-wise impurity scores when evaluating a data split. Negative values here will build more invariant trees.

Read more: https://en.wikipedia.org/wiki/Smooth_maximum

### Gamma
Varies over the interval [0,1]. Indicates weight placed on the  era splitting criterion.

### Blama
Varies over the interval [0,1]. Indicates weight placed on the directional era splitting criterion.

### Vanna
Varies over the interval [0,1]. Indicates weight placed on the original splitting criterion.

Behind the scenes, this is for formula which creates a linear combination of the split criteria. Usually we just set one of these to 1 and leave the other at zero.
```python
gain = gamma * era_split_gain + blama * directional_era_split_gain + vanna * original_gain
```

# Complete (New Updated) Code Notebook Examples Available here:

https://github.com/jefferythewind/era-splitting-notebook-examples

# Citations:

````
@misc{delise2023era,
      title={Era Splitting}, 
      author={Timothy DeLise},
      year={2023},
      eprint={2309.14496},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
````

This code was forked from the official scikit-learn repository and is currently a stand-alone version. All community help is welcome for getting these ideas part of the official scikit learn code base or even better, incorporated in the LightGBM code base.

https://scikit-learn.org/stable/about.html#citing-scikit-learn
