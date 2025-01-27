import mlflow 
import os 
import yaml
import sklearn
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , mean_squared_error , mean_absolute_error , classification_report , confusion_matrix

params=yaml.safe_load(open('params.yaml'))

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/DevManpreet5/end_end_ML_pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="DevManpreet5"
os.environ["MLFLOW_TRACKING_PASSWORD"]=" "


def test_model(dataset_path,model_path):
    data=pd.read_csv(dataset_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/DevManpreet5/end_end_ML_pipeline.mlflow")
    mlflow.set_experiment("evaluating model")


    with mlflow.start_run():
      
        print("testing started")
        best_model=pickle.load(open(model_path,'rb'))
        y_pred=best_model.predict(X)
        accuracy=accuracy_score(y_pred,y)
        mlflow.log_metric("accuracy",accuracy)

        cm=confusion_matrix(y,y_pred)
        cr=classification_report(y,y_pred)
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        print("evaluating done")




if __name__=="__main__":
    test_model(params['train']['input'],params['train']['model'])
    
