from pipeline.pipeline_train_test import PipelineScorerTraining
import pandas as pd
from joblib import dump, load
#  import cupy as cp
import torch
import time


class EntrenamientoProducto():
    def __init__(self):
        self.resultados_ = None

    def pipeline_entrenamiento_guardar(self):
        """
        info:
        """
        PST = PipelineScorerTraining()
        PST.train_pipeline(df_texto_train,
                           numero_features, maximum_df, min_df,
                           numero_de_componentes, max_iter,
                           beta_loss, solver, alpha,
                           l1_ratio, thresh_percentile)
        PST.test_pipeline(df_texto_test)
        score = PST.scorer(df_texto_test)
        print(score)


if __name__ == "__main__":

    s = time.time()

    df_entrenamiento = load(
        "./data/entrenamiento/datos_training_n_eval_5_sample.pkl")

    df_entrenamiento = df_entrenamiento.sample(1000)

    df_texto_train = df_entrenamiento
    df_texto_test = df_entrenamiento

    # Parametros
    # TFidf
    min_df = 5
    # NMF
    max_iter = 150
    beta_loss = 'kullback-leibler'
    solver = 'mu'
    alpha = .1
    l1_ratio = .5
    thresh_percentile = 60

    # Parametros obtenidos de la búsqueda de hiperparámetros
    numero_de_componentes = 150
    numero_features = 150
    maximum_df = 0.5

    # entrenamiento

    EP = EntrenamientoProducto()
    EP.pipeline_entrenamiento_guardar()

    #  cp.cuda.Stream.null.synchronize()
    e = time.time()
    print("T(s)")
    print(e - s)
