import pickle
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
pd.options.mode.chained_assignment = None  # default='warn'


class ScorePipelineCoincidencias():
    """
    Obtiene cosine similarity entre los vectores sugeridos y los reales
    """

    def __init__(self):
        self.df_texto_topico_ = None
        self.df_evaluadores_reco_ = None
        self.metrica_mean = None
        self.df_proyectos_eval_ = None

    def cosine_similarity_metric(self, df_evaluador_topico_,
                                 df_texto_topico_,
                                 df_proyectos_eval_,
                                 evaluadores_a_sugerir):
        """
        obtiene la metrica para el conjunto
        de prueba

        Devuelve una lista con la metrica por proyecto evaludo

        """
        #  df_evaluador_topico_

        df_evaluador_topico_.reset_index(drop=False, inplace=True)
        usuario_entreanado = df_evaluador_topico_["CVU"].tolist()
        rcea_entreanado = df_evaluador_topico_["CVU"].tolist()

        # avealuar
        df_texto_topico_.drop_duplicates("ID_PROYECTO", inplace=True)

        id_proyecto_index = df_texto_topico_["ID_PROYECTO"].tolist()
        # cos simi
        print(df_evaluador_topico_.head(2))
        print(df_texto_topico_.head(2))
        cos_simi = cosine_similarity(
            df_evaluador_topico_.iloc[:, 1:], df_texto_topico_.iloc[:, :-3])
        df_cos_simi = pd.DataFrame(cos_simi)

        # obtener los top N evaluadores sugeridos por proyecto
        df_evaluadores_recomendados = pd.DataFrame(
            index=[i for i in range(len(id_proyecto_index))],
            columns=["evaluador_reco_" + str(number)
                     for number in range(evaluadores_a_sugerir)])
        len_list = []
        for proyecto in range(df_cos_simi.shape[1]):

            list_each_proyecto = df_cos_simi.iloc[:, proyecto].tolist()

            top_5_idx = np.argsort(
                list_each_proyecto)[- evaluadores_a_sugerir:][::-1].tolist()
            #  top_5_idx = [i for i in top_5_idx]

            df_evaluadores_recomendados.iloc[proyecto, :] = [
                usuario_entreanado[pos] for pos in top_5_idx]

        df_evaluadores_recomendados["ID_PROYECTO"] = id_proyecto_index

        df_evaluadores_recomendados.set_index(
            "ID_PROYECTO").reset_index(inplace=True, drop=False)

        #  Obtenemos la lista de CVUs para cada id_proyecto
        self.df_proyectos_eval_ = df_proyectos_eval_

        self.df_proyectos_eval_["CVU"] = self.df_proyectos_eval_[
            "CVU"].astype(int).astype(str)
        self.df_proyectos_eval_ = self.df_proyectos_eval_[[
            "ID_PROYECTO", "CVU"]]
        self.df_proyectos_eval_ = self.df_proyectos_eval_.groupby(
            'ID_PROYECTO')['CVU'].apply(
            lambda x: "[%s]" % ', '.join(x))

        df_proy_usuarios = pd.DataFrame(
            index=self.df_proyectos_eval_.index, columns=["USUARIOS"])
        df_proy_usuarios.loc[:, "USUARIOS"] = self.df_proyectos_eval_
        df_proy_usuarios.reset_index(drop=False, inplace=True)

        # merge textos con conjunto de prueba
        df_evaluadores_recomendados = df_evaluadores_recomendados.merge(
            df_proy_usuarios, on="ID_PROYECTO")

        #  Obtenemos las coincidencias entre los evaluadores reales y sugeridos
        matches_list = []
        for proyecto in range(len(df_evaluadores_recomendados)):
            #  arreglamos los espacios vacios en los CVU para no
            #  tener problemas en la comparacion de strings
            lista_evaluadores_reales = df_evaluadores_recomendados.iloc[
                proyecto, -1].replace(
                ']', '').replace(
                " ", "").replace('[', '').replace('"', '').split(",")

            lista_matches = len([ev_recom
                                for ev_recom
                                in df_evaluadores_recomendados.iloc[
                                    proyecto, :-2].tolist()
                                if str(ev_recom) in lista_evaluadores_reales])

            lista_eval_len = len(lista_evaluadores_reales)

            #  obtenemos la metrica de coincidencias, explicada en el readme
            metrica = lista_matches/lista_eval_len
            matches_list.append(metrica)

        df_evaluadores_recomendados["metrica"] = matches_list
        self.metrica_mean = df_evaluadores_recomendados["metrica"].mean()

        return self.metrica_mean

# class ScorePipeline():
#     """
#     obtiene cosine similarity entre los vectores sugeridos y los reales
#     """
#
#     def __init__(self):
#         self.df_evaluador_topico_ = None
#         self.df_texto_topico_ = None
#         self.df_evaluadores_reco_ = None
#         self.df_proyectos_eval_ = None
#
#     def vector_mean_var(self, df, lista_row):
#         """
#         Obtiene el vector promedio y la varianza de un conjunto de vectores
#         y el promedio de la similitud de coseno entre el vector promedio
#         y sus componentes
#         """
#         print(df)
#         vector_array = [df.iloc[pos, 2:].values for pos in lista_row]
#         mean = np.mean(vector_array, axis=0)
#         var = np.var(cosine_similarity(vector_array))
#         dis = np.mean([cosine_similarity(mean.reshape(1, -1),
#                                          vector.reshape(1, -1))
#                       for vector in vector_array])
#         return mean, var, dis
#
#     def vector_mean(self, df_evaluador_topico, df_texto_topico):
#         """
#
#
#
#
#         """
#         print("...score")
#         # evaluador topico
#         print("df_evaluador_topico")
#         print(df_evaluador_topico)
#         print("df_texto_topico")
#         print(df_texto_topico)
#         self.df_evaluador_topico_ = df_evaluador_topico
#
#         self.df_evaluador_topico_ .reset_index(drop=False, inplace=True)
#         # proyecto a asignar eval
#
#         self.df_texto_topico_ = df_texto_topico
#         self.df_texto_topico_.drop_duplicates(subset="ID_PROYECTO",
#                                               inplace=True)
#         id_proyecto_index = self.df_texto_topico_["ID_PROYECTO"].tolist()
#
#         # df para guardar resultados
#         self.df_evaluadores_reco_ = pd.DataFrame(
#             index=[i for i in range(len(id_proyecto_index))],
#             columns=["eval_reco_mean", "eval_reco_var", "eval_reco_dis",
#                      "eval_actual_mean", "eval_actual_var",
#                      "cos_sim", "metrica"])
#         # print(df_evaluador_topico.shape)
#         # print(self.df_texto_topico_.shape)
#         cos_simi = cosine_similarity(
#             df_evaluador_topico.iloc[:, 1:],
#             self.df_texto_topico_.iloc[:, : -1])
#         df_cos_simi = pd.DataFrame(cos_simi)
#
#         print("...cosine")
#         for proyecto in range(df_cos_simi.shape[1]):
#
#             list_each_proyecto = df_cos_simi.iloc[:, proyecto].tolist()
#
#             top_5_idx = np.argsort(list_each_proyecto)[-5:][::-1]
#
#             vector_array = [df_evaluador_topico.iloc[pos, 2:].values
#                             for pos in top_5_idx]
#             mean = np.mean(vector_array, axis=0)
#             var = np.var(cosine_similarity(vector_array))
#             metrica = np.mean(
#                 cosine_similarity(mean.reshape(1, -1), vector_array)[0])
#
#             self.df_evaluadores_reco_.loc[proyecto, "eval_reco_mean"] = mean
#             self.df_evaluadores_reco_.loc[proyecto, "eval_reco_var"] = var
#             self.df_evaluadores_reco_.loc[proyecto, "metrica"] = metrica
#
#         self.df_evaluadores_reco_["ID_PROYECTO"] = id_proyecto_index
#         self.df_evaluadores_reco_.set_index("ID_PROYECTO").reset_index(
#                             inplace=True, drop=False)
#
#     def verificar_pertenencia(self):
#         """
#
#
#
#
#         """
#         #  self.df_proyectos_eval_ = load("./data/data_convocatorias.pkl")
#         self.df_proyectos_eval_ = self.df_proyectos_eval_.reset_index(
#             drop=True)
#         self.df_proyectos_eval_ = self.df_proyectos_eval_[[
#                             "ID_PROYECTO", "CVU"]]
#
#         lista_de_usuarios = self.df_proyectos_eval_["CVU"].unique()
#         id_proyecto_list = self.df_evaluadores_reco_["ID_PROYECTO"]
#
#         # obtenemos el vector promedio y la varianza de su cosine sim
#         self.df_evaluadores_reco_.set_index("ID_PROYECTO",  inplace=True)
#
#         for id_proyecto in id_proyecto_list:
#             lista_evaluadores = self.df_proyectos_eval_[
#                 self.df_proyectos_eval_["ID_PROYECTO"] ==
#                 id_proyecto]["CVU"].values.tolist()
#
#             lista_evaluadores = self.df_evaluador_topico_[
#                 self.df_evaluador_topico_["CVU"].isin(lista_evaluadores)].index
#
#             vector_array = [self.df_evaluador_topico_.iloc[pos, 2:].values
#                             for pos in lista_evaluadores]
#
#             if vector_array == []:
#                 mean = None
#                 var = None
#
#             else:
#                 mean = np.mean(vector_array, axis=0)
#                 var = np.var(cosine_similarity(vector_array))
#
#             self.df_evaluadores_reco_.loc[
#                 id_proyecto, "eval_actual_mean"] = mean
#             self.df_evaluadores_reco_.loc[
#                 id_proyecto, "eval_actual_var"] = var
#
#         self.df_evaluadores_reco_.dropna(
#             subset=['eval_reco_var', 'eval_actual_mean'], inplace=True)
#         self.df_evaluadores_reco_.reset_index("ID_PROYECTO", inplace=True)
#
#     def cos_sim_entre(self):
#         """
#         computa el coseno entre el promedio de los
#         evaluadores sugeridos y actuales
#         """
#         cos_sim = []
#         for row in range(len(self.df_evaluadores_reco_)):
#             cos_sim.append(
#                 cosine_similarity(
#                     self.df_evaluadores_reco_[
#                         "eval_reco_mean"][row].reshape(1, -1),
#                     self.df_evaluadores_reco_[
#                         "eval_actual_mean"][row].reshape(1, -1)
#                                  )[0][0])
#
#         self.df_evaluadores_reco_["cos_sim"] = cos_sim
#
#         cos_sim_mean = self.df_evaluadores_reco_["cos_sim"].mean()
#         metrica_mean = self.df_evaluadores_reco_["metrica"].mean()
#
#         return cos_sim_mean, metrica_mean
#
