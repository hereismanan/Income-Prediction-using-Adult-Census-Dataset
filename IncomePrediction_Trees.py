import pandas as pd


columns=[ 'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df=pd.read_csv(url,header=None,names=columns,na_values="?",skipinitialspace=True)

print(df.shape)
#print(df.head())

#print(df.info())

#print(df.isnull().sum())

df['income'].value_counts(normalize=True)


df['workclass']=df['workclass'].fillna("Unknown")
df['occupation']=df['occupation'].fillna("Unknown")
df['native-country']=df['native-country'].fillna("Unknown")

print(df.isnull().sum())

df=df.drop(columns=['fnlwgt'])

print(df.columns)

categorical_cols=df.select_dtypes(include=['object']).columns.tolist()

numeric_cols=df.select_dtypes(include=['int64','float64']).columns.tolist()


df['income']=df['income'].apply(lambda x:1 if x.strip()=='>50K' else 0)

print(df['income'].value_counts(normalize=True))


from sklearn.preprocessing import OneHotEncoder

categorical_cols.remove('income')
ohe=OneHotEncoder(sparse_output=False,handle_unknown="ignore")

enhanced=ohe.fit_transform(df[categorical_cols])

from sklearn.model_selection import train_test_split

X=df.drop(columns=['income'])
y=df['income']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

#stratify=>ensures ration is preserved in both train and test split


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder

preprocessor=ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_cols),
        ('nums','passthrough',numeric_cols)
    ]
)

dt_pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',DecisionTreeClassifier(random_state=42))
])


dt_pipeline.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=dt_pipeline.predict(X_test)

print("Accuracy of Decision Tree:",accuracy_score(y_test,y_pred))


from sklearn.ensemble import RandomForestClassifier

rf_pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train,y_train)

y_pred_rf=rf_pipeline.predict(X_test)

print("Accuracy of Random Forest:",accuracy_score(y_test,y_pred_rf))



from xgboost import XGBClassifier
xgb_pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ))
])

xgb_pipeline.fit(X_train,y_train)

y_pred_xgb=xgb_pipeline.predict(X_test)

print("Accuracy of XGBoost:",accuracy_score(y_test,y_pred_xgb))

import joblib

joblib.dump(dt_pipeline,'decision_tree_pipeline.pkl')
joblib.dump(rf_pipeline,'random_forest_pipeline.pkl')
joblib.dump(xgb_pipeline,'xgboost_pipeline.pkl')

print("All model saveds")