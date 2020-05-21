from pipeline.pipeline_train_test import PipelineScorerTraining
from pipeline.dataset_split import SplitTrainTest
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class GridSearch():

    def __init__(self):
        self.resultados_ = None

    def grid_search(self):
        """
        Ejecuta el grid grid_search
        """
        iteracion = 1
        self.resultados_ = []
        iter_total = len(n_components_grid)*len(n_features_grid) * \
            len(max_df_grid)*len(l1_ratio_grid)

        for numero_de_componentes in n_components_grid:
            for numero_features in n_features_grid:
                for maximum_df in max_df_grid:
                    for min_df in min_df_grid:
                        for l1_ratio in l1_ratio_grid:
                            for beta_loss in beta_loss_grid:
                                for solver in solver_grid:
                                    for alpha in alpha_grid:
                                        for thresh_percentile in thresh_percentile_grid:

                                            print("\n         ------------Iteracion {}/{}-----------\n".format(
                                                iteracion, iter_total))
                                            PST = PipelineScorerTraining()
                                            PST.train_pipeline(df_texto_train,
                                                               numero_features, maximum_df, min_df,
                                                               numero_de_componentes, max_iter,
                                                               beta_loss, solver, alpha,
                                                               l1_ratio, thresh_percentile)
                                            PST.test_pipeline(df_texto_test)
                                            score = PST.scorer(df_texto_test,
                                                               evaluadores_a_sugerir=5)

                                            self.resultados_.append(
                                                {"numero_de_componentes": numero_de_componentes,
                                                 "numero_features": numero_features,
                                                 "maximum_df": maximum_df,
                                                 "l1_ratio": l1_ratio,
                                                 "score": score})
                                            print(self.resultados_)

                                            dump(self.resultados_,
                                                 "./trained_models/grid_search/grid_search.pkl")
                                            iteracion = len(self.resultados_) + 1

    def score_matrix_plot(self):
        """
        Grafica los resultados del grid search como una
        matrix matrix plot
        """

        columns_grid_search = ['numero_de_componentes', 'numero_features',
                               'maximum_df', 'l1_ratio', 'score_num']
        # convierte el diccionario en DF
        df_resultados = pd.DataFrame(index=[i for i in
                                            range(len(self.resultados_))],
                                     columns=columns_grid_search)

        for row in range(len(df_resultados)):
            df_resultados.iloc[row, :] = [val for val in
                                          self.resultados_[:][row].values()]
        df_resultados = df_resultados.apply(pd.to_numeric)

        # Normaliza el DF para mejorar la visualizacion
        x = df_resultados.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_resultados = pd.DataFrame(x_scaled)
        df_resultados.columns = columns_grid_search

        # Compone la figura
        f = plt.figure(figsize=(20, 10))
        plt.matshow(df_resultados, fignum=f.number)
        plt.xticks(range(df_resultados.shape[1]),
                   df_resultados.columns, fontsize=10, rotation=90)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.show()
        plt.savefig('./trained_models/grid_search/grid_search.png')


if __name__ == "__main__":

    # leemos el archivo de entrenamiento
    #  df_entrenamiento = load(
    #     "./data/entrenamiento/datos_training_n_eval_5_sample.pkl")
    df_entrenamiento = pd.read_csv(
         "./data/entrenamiento/data_training.csv")
    # dividimos el conjunto en prueba y test
    STT = SplitTrainTest()
    df_texto_train, df_texto_test = STT.split_train_test(
                                df=df_entrenamiento,
                                thresh_min_frecuencia_eval=5,
                                thresh_max_frecuencia_eval=-1,
                                porcentaje_muestra=1)
    df_texto_train, df_texto_test = df_entrenamiento, df_entrenamiento
    print("Tamaño de los conjuntos:")
    print("df_train: {}\ndf_test: {}".format(df_texto_train.shape[0],
                                             df_texto_test.shape[0]))

    # Parametros fijos del modelo
    max_iter = 2

    # Lista de parámetros del grid search
    # parametros tfidf
    n_features_grid = [50]
    max_df_grid = [.3]
    min_df_grid = [0.0]
    thresh_percentile_grid = [0.0]
    # parametros NMF
    beta_loss_grid = ['kullback-leibler']  # ‘frobenius’, ‘kullback-leibler’,
    solver_grid = ["mu"]  # 'cd', 'mu'
    l1_ratio_grid = [0]
    alpha_grid = [0]
    n_components_grid = [5]
    l1_ratio_grid = [0.5]

    # Grid search
    GS = GridSearch()
    GS.grid_search()
    GS.score_matrix_plot()
