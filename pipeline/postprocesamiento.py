import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from joblib import dump, load

stemmer = SnowballStemmer('spanish')
warnings.filterwarnings("ignore")


class PostProcesamiento():
    """
    aplica self.tfidf_matrix_ a textos para obtener matriz de
    frecuencia de término
    """

    # def __init__(self):
    # self.self.df_topicos__ = None
    # self.self.tfidf_matrix__matrix_ = None
    # self.df_cleaned_ = None
    # self.self.tfidf_matrix__matrix_test_ = None
    # self.self.df_topicos__ = None
    # # NMF
    # self.topic_model_ = None
    # self.self.topic_data_ = None
    # self.df_ = None
    # self.df_cleaned_ = None

    def topico_a_texto(self, status="test"):

        """
        Añade el vector de topicos a partir del producto punto entre la matriz
        de topicos de  NMF y la matrz de pesos de tfidf_matrix_ de cada
        documento.


        Parameters:
        -----------
        self.df_topicos_: matrix or sparse array
            matriz de tópicos generados por NMF.

        self.tfidf_matrix_ : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.

        self.topic_data: NMF model
            Matriz NMF
        Returns:
        --------
        self.dataframe_values_: df
            dataframe con topico por columna para cada texto
        """

        print("...topico_a_texto")

        lista_topicos = self.df_topicos_.index.tolist()
        self.dataframe_values_ = pd.DataFrame(
            columns=lista_topicos,
            index=[row for row in range(self.tfidf_matrix_[:].shape[0])])

        topics_results = []

        # computa el producto punto
        for i_doc in range(self.tfidf_matrix_[:].shape[0]):
            valor_topico = [
                np.dot(
                    self.tfidf_matrix_[i_doc].todense().tolist()[0],
                    self.topic_data[topic_id]["value"])
                for topic_id in range(len(self.topic_data))]  # pesos topicos

            self.dataframe_values_.iloc[i_doc, :] = valor_topico
            topics_results.append(str(valor_topico))

            # guardamos el vector generado para elastic solo en enntrenamiento
        if status == "train":
            self.df_texto_eval["topic_vector"] = topics_results
            # TF_idf_vector
            self.df_texto_eval["tfidf"] = [
                self.tfidf_matrix_.toarray()[row]
                for row in range(len(self.df_texto_eval))]

            dump(self.df_texto_eval, './trained_models/df_texto_eval.pkl')

        # return self.dataframe_values_

    def texto_a_evaluador(self, status="test"):

        """
        Si se está entrenando el modelo, asigna el vector de topicos de cada
        texto revisado por un evaluador y devuelve el vector promedio de todos
        los textos que ha evaluado aplicando un groupby.

        Si se estan generanndo los vectores para textos nuevos, asigna la
        etiqueta ID_proyecto al vector.

        Parameters:
        -----------
        self.dataframe_values_: df
            dataframe con topico por columna para cada texto
        Returns:
        --------
        self.dataframe_values_: df
            dataframe con topico por columna para cada texto con etiquetas
            asignadas
        """

        print("...texto_a_evaluador")

        index_self.dataframe_values_ = self.dataframe_values_.columns.tolist()[:]

        self.dataframe_values_[index_self.dataframe_values_] = self.dataframe_values_[
            index_self.dataframe_values_].apply(
            pd.to_numeric, errors='coerce').reset_index(drop=True)

    # informacion de proyecto y evaluador
        self.df_texto_eval = load('./trained_models/self.df_texto_eval.pkl')

        self.dataframe_values_["ID_PROYECTO"] = self.df_texto_eval[
            "ID_PROYECTO"].reset_index(drop=True)
        # poner campos en primera posicion
        self.dataframe_values_ = self.dataframe_values_.set_index(
            ["ID_PROYECTO"]).reset_index(drop=False)

        if status == "train":
            # merge con evaluadores
            df_info_eval = pd.read_csv(
                "./data/data_training.csv").reset_index(drop=True)
            df_info_eval = df_info_eval[["ID_PROYECTO", "USUARIO",
                                        "CVU", "CVE_RCEA"]]

            self.dataframe_values_ = df_info_eval.merge(self.dataframe_values_,
                                                        on="ID_PROYECTO",
                                                        how="inner")

        # groupby de los vectores por evaluador
            topics_evaluador = self.dataframe_values_.groupby(
                ["CVE_RCEA", "USUARIO"])[index_self.dataframe_values_[3:]].mean()

            dump(topics_evaluador,
                 './trained_models/topics_evaluador.pkl')
            return topics_evaluador
        else:
            dump(self.dataframe_values_,
                 './trained_models/topicos_port_texto_test.pkl')
            return self.dataframe_values_


if __name__ == '__main__':
    df_texto_eval = pd.read_csv(
        "../data/data_training.csv").reset_index(drop=True)
    df_texto_eval = df_texto_eval.reset_index(drop=True)
    df_texto_eval = df_texto_eval.drop_duplicates(
        subset=["ID_PROYECTO", "NUMERO_CONVOCATORIA", "ANIO"], keep="last")
    df_texto_eval = df_texto_eval.sample(400)
    texto = df_texto_eval["DESCRIPCION_PROYECTO"]
    len(texto)

    n_features = 512  # number of max words
    n_top_words = 30  # words per topic
    doc_similarity_thr = 0.15
    max_df = .15
    min_df = 5

    TPT = TrainingPipelineTfidf()
    TPT.tfidf_train(texto, max_df, min_df, n_features)

    n_components = 50
    max_iter = 50

    TPT.train_nmf(TPT.tfidf_matrix_, n_components,
                  solver='mu', max_iter=200, alpha=.1, l1_ratio=.5)

    TPT.vocabulario_nmf()
    print(TPT.df_topicos_)

    thresh_percentile = 95

    TPT.filtro_vactores_nmf(thresh_percentile)
