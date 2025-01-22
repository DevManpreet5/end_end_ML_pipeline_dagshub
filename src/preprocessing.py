import os 
import sys 
import pandas as pd 
import yaml

params=yaml.safe_load(open("params.yaml"))['preprocessing']



def preprocessingfxn(input_path,output_path):
    df=pd.read_csv(input_path)
    print("data loaded ")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path,index=False)
    print("data preprocessed")

if __name__=="__main__":
    preprocessingfxn(params['input'],params['output'])
    
