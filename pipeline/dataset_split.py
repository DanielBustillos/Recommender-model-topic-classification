import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample


class SplitTrainTest():
    """
    aplica self.tfidf_matrix_ a textos para obtener matriz de
    frecuencia de término
    """

    def unir_dataframes(self, df_list):
        """
        Une una lista de dataframes en uno solo.


        Parameters:
        -----------
        df_list: list
                lista donde cada entrada es un DF.

        Returns:
        --------
        df_ dataframe
            dataframe del resultado de la union.
        """
        df = df_list[0].append(df_list[1])
        for i in range(2, len(df_list)):
            df = df.append(df_list[i].reset_index(drop=True))
        df.reset_index(inplace=True, drop=True)
        return df

    def split_train_test(self, df, thresh_min_frecuencia_eval,
                         thresh_max_frecuencia_eval, porcentaje_muestra):
        """
        Divide el conjunto de datos de los proyectos en dos, uno de prueba
        y  otro de entrenamiento. En el conjunto de prueba se almacena un
        proyecto por evaluador, en el de entrenamiento se almacenan hasta
        5 textos por evaluador y un minimo de 2.


        Parameters:
        -----------
        df: Dataframe
            datos de los proyectos ya limpios.

        threshold_frecuencia_eval: int
            numero minimo de proyectos por evaluador.

        porcentaje_muestra: float
            porcentaje de la muestra del conjunto total de datos.

        Returns:
        --------
        df_test: dataframe
            dataframe de prueba con un texto por evaluador.

        df_train: dataframe
            dataframe de prueba con un una cantidad de textos definida por
            threshold_frecuencia_eval por evaluador.
        """

        # veamos que thresh_max_frecuencia_eval > thresh_min_frecuencia_eval
        if thresh_max_frecuencia_eval + 1 > thresh_min_frecuencia_eval:
            raise Exception(
                "Error: Frecuencia minima igual a frecuencia maxima")

        #  Primero obtengamos la liste de CVU's que se repiten al menos 3 veces

        frecuencia_cvu = pd.DataFrame(df['CVU'].value_counts())
        lista_cvus = frecuencia_cvu[
                      frecuencia_cvu["CVU"] >= thresh_min_frecuencia_eval
                      ].index.tolist()

        #  Obtenemos una muestra de los evaluadores con el fin de alivianar
        #  el proceso de grid search.
        num_evals = len(lista_cvus)
        lista_cvus = sample(lista_cvus, round(num_evals * porcentaje_muestra))

        #  filtramos el df
        df = df[df["CVU"].isin(lista_cvus)]

        #  dividimos el df en train, test
        lista_coincidencia_cvu_train = []
        lista_coincidencia_cvu_test = []

        for CVU in lista_cvus:
            coincidencia_cvu_train = df[
                df["CVU"] == CVU].tail(thresh_max_frecuencia_eval)
            coincidencia_cvu_test = df[
                df["CVU"] == CVU].head(1)

            # guardamos cada df en una lista
            lista_coincidencia_cvu_train.append(coincidencia_cvu_train)
            lista_coincidencia_cvu_test.append(coincidencia_cvu_test)

        # juntamos los dataframes en uno:
        df_train = self.unir_dataframes(df_list=lista_coincidencia_cvu_train)
        df_test = self.unir_dataframes(df_list=lista_coincidencia_cvu_test)

        return df_train, df_test
