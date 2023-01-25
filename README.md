# Dynamic Ensemble Selection 

### Installation 

```bash
pip install infodeslib
```

### Example 

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

2. Train the models (pool): 

```python 
for i in range(len(model_pool)): 
    model_pool[i].fit(X_pool[feature_sets[i]], y_pool)
    
    acc = round(model_pool[i].score(X_dsel[feature_sets[i]], y_dsel), 3) 
    print("[DSEL] Model {} acc: {}".format(i, acc)) 

    acc = round(model_pool[i].score(X_test[feature_sets[i]], y_test), 3)  
    print("[Test] Model {} acc: {}".format(i, acc))  
```

3. Usage of our library: 

```python
from infodeslib.des.knorau import KNORAU 

colors = {0: 'red', 1: 'green'}  

# initializing 
knorau = KNORAU(model_pool, feature_sets, k=7,  
                DFP=True, plot=False, X_dsel=X_dsel, y_dsel=y_dsel, colors=colors)
``` 

4. Testing 

```python 
from tqdm import tqdm 

preds = [] 

for i in tqdm(range(X_test.shape[0])):
    query = X_test.iloc[[i]] 
    
    pred, conf = knorau.predict(query)
    preds.append(pred) 
    

acc = round(accuracy_score(y_test, preds), 3) 
print("[Test] acc: {}".format(acc))
```

5. Explainability 

```python 
colors = {0: 'red', 1: 'green'}  

## Make plot=True 
knorau = KNORAU(model_pool, feature_sets, k=7,  
                DFP=True, plot=True, X_dsel=X_dsel, y_dsel=y_dsel, colors=colors)

query = X_test.iloc[[5]]

knorau.predict(query, y_test.iloc[5])
```

