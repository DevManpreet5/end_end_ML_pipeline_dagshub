import mlflow 
import os 
import yaml
import sklearn
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import dagshub
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score , mean_squared_error , mean_absolute_error , classification_report , confusion_matrix

params=yaml.safe_load(open('params.yaml'))

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/DevManpreet5/end_end_ML_pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="DevManpreet5"
os.environ["MLFLOW_TRACKING_PASSWORD"]=" "

def hyperparameter_tuning(grid,X_train,y_train):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

def train_model(dataset_path,model_path):
    data=pd.read_csv(dataset_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/DevManpreet5/end_end_ML_pipeline.mlflow")


    with mlflow.start_run():
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        print("training started")
        grid_search=hyperparameter_tuning(param_grid,X_train,y_train)
        best_model=grid_search.best_estimator_
        y_pred=best_model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        signature=infer_signature(X_train,best_model.predict(X_train))

  
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimatios",grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        modellog=mlflow.sklearn.log_model(
            best_model,"bestmodel",signature=signature,registered_model_name="best_model"
        )

        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))
        print(f"Model saved to {model_path}")



if __name__=="__main__":
    train_model(params['train']['input'],params['train']['model'])
    
