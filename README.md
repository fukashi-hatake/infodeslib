# Dynamic Ensemble Selection 

### Installation 

```bash
pip install infodeslib
```

# A little example 

Loading necessary libraries and dataset:  

```python
import warnings
warnings.filterwarnings('ignore') 

import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import accuracy_score 

## Load simple open dataset 
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target 

```

Split the dataset into training, validation for DES (DSEL) and testing. 

```python
X = df.drop(['target'], axis=1) 
y = df.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_pool, X_dsel, y_pool, y_dsel   = train_test_split(X_train, y_train, test_size=0.30, random_state=42) 

```

1. Models and Feature sets Generation 

```python
model1 = SVC(probability=True, random_state=42)
model2 = RandomForestClassifier(random_state=42) 
model3 = KNeighborsClassifier() 

feature_set1 = data.feature_names[:10] 
feature_set2 = data.feature_names[10:20]
feature_set3 = data.feature_names[20:]

model_pool = [model1, 
              model2, 
              model3]

feature_sets = [feature_set1, 
                feature_set2, 
                feature_set3] 
```