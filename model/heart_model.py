#%%
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

#%%
df = pd.read_csv(r"C:\Users\Rawan Alamily\Downloads\McSCert Co-op\tabnet-heart\data\life-heart.csv")
df['target'] = np.where(df['heartDisease']=='Yes', 1, 0)
df = df.drop(columns=['heartDisease'])
df.head()
#%%
yn = lambda x: 1 if x=='Yes' else 0
male = lambda x: 1 if x=='Male' else 0
def age(x):
    y = int(x[0:2])
    return y
def diabetes(x):
    if x=='Yes':
        y=1
    elif x=='No':
        y=0
    else:
        y=2
    return y
def genHealth(x):
    if x=='Very good':
        y=0
    elif x=='Good':
        y=1
    elif x=='Excellent':
        y=2
    elif x=='Fair':
        y=3
    else:
        y=4
    return y
#%%
def encode_strings(df):
    for i in range(df.iloc[:,:].shape[0]): 
        for j in range(1,4):
            df.iloc[i,j] = yn(df.iloc[i,j])
        df.iloc[i,6] = yn(df.iloc[i,6])
        df.iloc[i,7] = male(df.iloc[i,7])
        df.iloc[i,8] = age(df.iloc[i,8])
        df.iloc[i,9] = diabetes(df.iloc[i,9])
        df.iloc[i,10] = yn(df.iloc[i,10])
        df.iloc[i,11] = genHealth(df.iloc[i,11])
        for j in range(13,16):
            df.iloc[i,j] = yn(df.iloc[i,j])
    return df 
#%%
df = df[:10000,:]
df = encode_strings(df)    
#%%
df.head()
#%%
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.9*len(df))])
#%%
y_train = train.pop('target')
X_train = train
y_val = val.pop('target')
X_val = val
y_test = test.pop('target')
X_test = test
#%%

#%%
clf = TabNetClassifier()
clf.fit(
    X_train, y_train,
    eval_set=(X_val,y_val)
)
# %%
