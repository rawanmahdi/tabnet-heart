#%%
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

#%%
def load_df(path):
    df = pd.read_csv(path)
    df['target'] = np.where(df['heartDisease']=='Yes', 1, 0)
    df = df.drop(columns=['heartDisease'])
    return df
df = load_df(r"C:\Users\Rawan Alamily\Downloads\McSCert Co-op\tabnet-heart\data\life-heart.csv")
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
    df['smoking'] = df['smoking'].astype(str).astype(int)
    df['alcoholDrinking'] = df['alcoholDrinking'].astype(str).astype(int)
    df['stroke'] = df['stroke'].astype(str).astype(int)
    df['diffWalk'] = df['diffWalk'].astype(str).astype(int)
    df['sex'] = df['sex'].astype(str).astype(int)
    df['ageGroup'] = df['ageGroup'].astype(str).astype(int)
    df['diabetic'] = df['diabetic'].astype(str).astype(int)
    df['physicalActivity'] = df['physicalActivity'].astype(str).astype(int)
    df['overallHealth'] = df['overallHealth'].astype(str).astype(int)
    df['asthma'] = df['asthma'].astype(str).astype(int)
    df['kidneyDisease'] = df['kidneyDisease'].astype(str).astype(int)
    df['skinCancer'] = df['skinCancer'].astype(str).astype(int)
    return df 
#%%
df = df.iloc[:25000,:]
df = encode_strings(df)    
#%%
corr_mat = df.corr()
corr_mat
#%%
df = df.copy()
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
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train,y_train)
X_val_resampled, y_val_resampled= rus.fit_resample(X_val, y_val)
#%%
cat_idxs = [1,2,3,6,7,8,9,10,11,13,14,15]
# cat_dims = {1:2, 2:2, 3:2,
#            6:2, 7:2, 8:14, 9:2,
#            10:2, 11:5, 13:2, 
#            14:2, 15:2}
cat_dims = [2,2,2,2,2,14,2,2,5,2,2,2]
#%%
len(cat_idxs)==len(cat_dims)
#%%
clf = TabNetClassifier(cat_dims=cat_dims, 
                       cat_idxs=cat_idxs,
                       cat_emb_dim=15)
#%%
clf = TabNetClassifier()
#%%
X = np.array(X_val)
Y = np.array(y_val)
eval = [(X,Y)]
# print(eval)

# x = np.array([[1,1,1],[1,1,1]])
# y = np.array([0,1])
# eval = (x,y)
# print(eval)
for X, y in eval:
    print(X)
    print(y)
#%%
clf.fit(
    X_train=X_train.values, y_train=y_train.values, 
    eval_set=[(X_val.values, y_val.values)],
    eval_metric=['balanced_accuracy', 'accuracy'], 
    patience=0, 
    max_epochs=200,
    weights=1)
#%%
import matplotlib.pyplot as plt
plt.plot(clf.history['loss'])

#%%
plt.plot(clf.history['val_0_balanced_accuracy'])
plt.plot(clf.history['val_0_accuracy'])

#%%
for value in X_train:
    print((value))
# %%
(X_val,y_val)
# %%
