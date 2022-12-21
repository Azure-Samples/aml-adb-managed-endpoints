import mlflow
import json
import pandas as pd
import os
import xgboost as xgb
import time
import numpy as np


# Called when the deployed service starts
def init():
    global model
    global train_stats
    
    # Get the path where the deployed model can be found.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './model')
    
    # Load model
    model = mlflow.xgboost.load_model(model_path)

# Our sample payload (to be able to automatically generate swagger file)
input_sample = pd.DataFrame(data=[{"Idade": 21,
                                   "RendaMensal": 9703, 
                                   "PercentualUtilizacaoLimite": 1.0, 
                                   "QtdTransacoesNegadas": 5.0, 
                                   "AnosDeRelacionamentoBanco": 12.0, 
                                   "JaUsouChequeEspecial": 0.0, 
                                   "QtdEmprestimos": 1.0, 
                                   "NumeroAtendimentos": 100, 
                                   "TMA": 300, 
                                   "IndiceSatisfacao": 2, 
                                   "Saldo": 6438, 
                                   "CLTV": 71}])

# This is an integer type sample. Use the data type that reflects the expected result.
output_sample = np.array([0])

def run(data):
    try:
        print("receiving input_data....")
        print(data)

        #data = pd.DataFrame(json.loads(data))
        data = pd.read_json(data, orient = 'split')
        print(data)

        data_xgb = xgb.DMatrix(data)
        
        print("predicting....")
        result = model.predict(data_xgb)

        print("result.....")
        print(result)
    # You can return any data type, as long as it can be serialized by JSON.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
