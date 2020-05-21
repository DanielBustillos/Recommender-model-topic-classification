import pandas as pd
import numpy as np
#  import cupy as cp
import time
import sys
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from joblib import dump

import time
from tqdm import tqdm_gui
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class AlgoritmosPipeline():
    """
    aplica tfidf_matrix a textos para obtener matriz de frecuencia de
    término
    """

    def __init__(self):
        self.tfidf_vectorizer_ = None
        self.tfidf_matrix_ = None
        self.vocabulario_topicos_ = None
        self.nmf_matrix_ = None
        self.topics_texto_ = None
        self.topics_evaluador_ = None
        self.texto_ = None
        self.status = None

    def guardar_identificadores(self, df, status):
        """
        Guarda las columnas para identificar a los evaluadores y los proyectos
        después de aplicar Topic-Modelling.
        Si se está entrenando, guarda las columnas:
            "ID_PROYECTO", "CVU","NUMERO_CONVOCATORIA", "ANIO"
        Y elimina los duplicados de las columnas:
            ID_PROYECTO", "NUMERO_CONVOCATORIA", "ANIO"
        Si se esta probando el modelo, guarda las columnas:
            "ID_PROYECTO", "NUMERO_CONVOCATORIA", "ANIO", "CVU"
        """

        self.status = status
        if status == "test":
            self.topics_evaluador_ = df[["ID_PROYECTO", "NUMERO_CONVOCATORIA",
                                         "ANIO", "CVU"]]
        elif status == "produccion":
            self.topics_evaluador_ = df[["ID_PROYECTO", "NUMERO_CONVOCATORIA",
                                         "ANIO"]]
            self.status = "test"

        elif status == "train":
            self.df_info_proyectos_ = df
            df = df.drop_duplicates(
                      subset=["ID_PROYECTO", "NUMERO_CONVOCATORIA", "ANIO"],
                      keep="last")
            self.texto_ = df["DESCRIPCION_PROYECTO"]  # textos unicos

            df = df[["ID_PROYECTO", "CVU",
                    "NUMERO_CONVOCATORIA", "ANIO"]]
            self.topics_evaluador_ = df
        else:
            raise ValueError('guardar_identificadores. status incorrecto')

    def tfidf_vectorizador_train(self, max_df, min_df, n_features):

        """
        Genera el vocabulario y la matriz de pesos usando tf-idf

        Parameters:
        -----------
        texto: string a aplicar tfidf_matrix

        max_df : float in range [0.0, 1.0] or int (default=1.0)
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents,
            integer absolute counts. This parameter is ignored if vocabulary
            is not None.

        min_df : float in range [0.0, 1.0] or int (default=1)
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is
            also called cut-off in the literature.
            If float, the parameter represents a proportion of documents,
            integer absolute counts. This parameter is ignored if vocabulary
            is not None.

        n_features : int or None (default=None)
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.

        Returns:
        --------
        tfidf_vectorizer:  TfidfVectorizer fiteado

        tfidf_matrix : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.

        tfidf_vectorizer: list
            feature names de tfidf_vectorizer
        """

        print("...tfidf_vectorizador_train")

        # definimos las stop words
        with open("./pipeline/stop_words_spanish.txt", 'r') as f:
            stop_words_spanish = f.readlines()[0].split(" ")

        tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                           max_features=n_features,
                                           stop_words=stop_words_spanish)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.texto_)

        self.tfidf_vectorizer_ = tfidf_vectorizer
        dump(self.tfidf_vectorizer_,
             "./trained_models/modelos/tfidf_vectorizer.pkl")
        self.tfidf_matrix_ = tfidf_matrix

    def tfidf_vectorizador_test(self, texto, tfidf_vectorizer):

        """
        Transforma nuevos textos usando tfidf_vectorizer


        Parameters:
        -----------
        texto: list
            list of strings a aplicar transformación usando tfidf_vectorizer

        tfidf_vectorizer : model_tfidf


        Returns:
        --------
        tfidf_vectorizador_test: matrix
            matriz de tfidf_matrix para nuevos textos.
        """

        print("...tfidf_vectorizador_test")

        tfidf_matrix_test = tfidf_vectorizer.transform(texto)
        # return tfidf_vectorizador_test
        self.tfidf_matrix_ = tfidf_matrix_test

    def train_nmf(self, n_components, beta_loss='frobenius',
                  solver='cd', max_iter=300, alpha=0.5, l1_ratio=.5,
                  init="nndsvd"):
        """
        Genera tópicos usando matríz de tfidf_matrix


        Parameters:
        -----------
        tfidf_matrix: string
            matriz de pesos generado en tfidf_matrix

        n_components : int or None
            Number of components, if n_components is not set all features
            are kept.

        beta_loss : float or string, default ‘frobenius’
            String must be in {‘frobenius’, ‘kullback-leibler’,
             ‘itakura-saito’}. Beta divergence to be minimized, measuring
             the distance between X and the dot product WH. Note that
             values different from ‘frobenius’ (or 2) and
             ‘kullback-leibler’ (or 1) lead to significantly slower fits.
             Note that for beta_loss <= 0 (or ‘itakura-saito’), the input
             matrix X cannot contain zeros. Used only in ‘mu’ solver.


        solver : ‘cd’ | ‘mu’
            Numerical solver to use: ‘cd’ is a Coordinate Descent solver.
            ‘mu’ is a Multiplicative Update solver.

        max_iter : integer, default: 100
            Maximum number of iterations before timing out.

        max_iter : integer, default: 200
            Maximum number of iterations before timing out.

        1_ratio : double, default: 0.
            The regularization mixing parameter, with 0 <= l1_ratio <= 1
            For l1_ratio = 0 the penalty is an elementwise L2 penalty
            (aka Frobenius Norm). For l1_ratio = 1 it is an elementwise L1
            penalty. For 0 < l1_ratio < 1, the penalty is a combination
            of L1 and L2.

        Returns:
        --------
        topic_model: matrix or sparse array
            matriz de tópicos generados por NMF.
        """
        print("...train_nmf")

        topic_model = NMF(n_components, random_state=123,
                          beta_loss=beta_loss,
                          solver=solver, max_iter=max_iter,
                          alpha=alpha, l1_ratio=l1_ratio, verbose=1)

        topic_model.fit(self.tfidf_matrix_)
        self.topic_model_ = topic_model

    def vocabulario_nmf(self):
        """
        Genera un diccionario con el index, palabra y peso de cada topico y lo
        pasa a un DF.

        Parameters:
        -----------
        topic_model: matrix
            matriz generado usando NMF matriz de pesos generado en tfidf_matrix

        feature_names: list
            nombre de palabras de diccionario.

        Returns:
        --------
        vocabulario_topicos_: matrix or sparse array
            matriz de tópicos generados por NMF.
        """
        print("...vocabulario_nmf")
        feature_names = self.tfidf_vectorizer_.get_feature_names()
        self.nmf_matrix_ = []

        for topic_idx, topic in enumerate(self.topic_model_.components_):
            index = [i for i in range(len(topic))]
            words = [feature_names[i] for i in index]
            value = [topic[i] for i in index]

            self.nmf_matrix_.append(
                {"index": index, "words": words, "value": value})

        filter_id = "topic-"
        self.vocabulario_topicos_ = pd.DataFrame(
            [t['value'] for t in self.nmf_matrix_])
        self.vocabulario_topicos_.index = [
            topico for topico in range(len(self.nmf_matrix_))]

        self.vocabulario_topicos_.columns = self.nmf_matrix_[0]['words']

    def filtro_vectores_nmf(self, thresh_percentile):
        """
        filtra los valores de peso de cada tópico de nmf segun el
        percentil del topico para cada topico

        Parameters:
        -----------
        thresh_percentile: float in [0,100]
            percentil para filtrar

        topic_data: NMF model
            Matriz NMF

        Returns:
        --------
        topic_data: NMF model
            Matriz NMF
        """

        print("...filtro_vectores_nmf")

        for topic in range(len(self.nmf_matrix_)):
            # valor de el filtro a partir de percentile
            thresh_filter = np.percentile(np.array(
                self.nmf_matrix_[topic]["value"]), thresh_percentile)

            values_filtered = [row if row > thresh_filter else 0
                               for row
                               in self.nmf_matrix_[topic]["value"]]
            self.nmf_matrix_[topic]["value"] = values_filtered
        dump(self.nmf_matrix_, "./trained_models/modelos/matriz_nmf.pkl")

    def producto_punto_topicos(self, nmf_matrix_=None):

        """
        Añade el vector de topicos a partir del producto punto entre la matriz
        de topicos de  NMF y la matrz de pesos de tfidf_matrix_ de cada
        documento.


        Parameters:
        -----------
        self.vocabulario_topicos_: matrix or sparse array
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

        print("...producto_punto_topicos")

        if self.status == "test":
            self.nmf_matrix_ = nmf_matrix_
        else:
            pass

        lista_topicos = [topico for topico in
                         range(len(self.nmf_matrix_))]

        # definimos el DF
        self.topics_texto_ = pd.DataFrame(
            columns=lista_topicos,
            index=[self.topics_evaluador_.index.tolist()])

        # computa el producto punto
        #  convertimos a nparray
        self.nmf_matrix_ = np.array(self.nmf_matrix_)
        self.tfidf_matrix_ = np.array(
            self.tfidf_matrix_.todense())

        if self.status != "test":
            self.nmf_matrix_ = [self.nmf_matrix_[iteracion]["value"]
                                for iteracion in range(
                                    self.nmf_matrix_.shape[0])]

        # convertimos los array a arrays en cupy
        #  almacenados en GPU para optimizar el cálculo
        if False: #torch.cuda.is_available():
            print("GPU disponible, computando producto punto en GPU.")

            tfidf_matrix_cupy = cp.asarray(self.tfidf_matrix_)
            nmf_matrix_cupy = cp.asarray(self.nmf_matrix_)
            # quitar el diccionario raro desde antes

            # producto punto usando cupy
            for i_doc in tqdm(range(tfidf_matrix_cupy.shape[0])):
                valor_topico = [
                    cp.dot(
                        tfidf_matrix_cupy[i_doc],
                        nmf_matrix_cupy[topic_id])
                    for topic_id in range(len(nmf_matrix_cupy))]
            # guardamos pesos topicos
                self.topics_texto_.iloc[i_doc, :] = valor_topico
            # al convertir las columnas se vuelven strings, convertir a float:
            self.topics_texto_ = self.topics_texto_.astype('float64')

            # volvemos de cupy a numpy
            self.nmf_matrix_ = cp.asnumpy(nmf_matrix_cupy)
            self.tfidf_matrix_ = cp.asnumpy(tfidf_matrix_cupy)

        else:
            print("GPU NO disponible, computando producto punto en CPU.")
            for i_doc in tqdm(range(self.tfidf_matrix_.shape[0])):
                valor_topico = [
                    np.dot(
                        self.tfidf_matrix_[i_doc],
                        self.nmf_matrix_[topic_id])
                    for topic_id in range(len(self.nmf_matrix_))]
            # guardamos pesos topicos
                self.topics_texto_.iloc[i_doc, :] = valor_topico

    def topicos_a_evaluador(self):
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

        print("...topicos_a_evaluador")

        # convierte a numerico
        columnas_numericas = self.topics_texto_.columns.tolist()
        self.topics_texto_[columnas_numericas] = self.topics_texto_[
            columnas_numericas].apply(
            pd.to_numeric, errors='coerce')  # .reset_index(drop=True)

        # pegamos la informacion de ID_PROYECTO, N convocatoria y ANIO
        self.topics_texto_["ID_PROYECTO"] = \
            self.topics_evaluador_["ID_PROYECTO"].tolist()
        self.topics_texto_["NUMERO_CONVOCATORIA"] = \
            self.topics_evaluador_["NUMERO_CONVOCATORIA"].tolist()
        self.topics_texto_["ANIO"] = \
            self.topics_evaluador_["ANIO"].tolist()
        # .reset_index(drop=True)

        #  pegamos con las conlumnas identificadoras según el caso
        if self.status == "train":

            #  nos interesa tener un vector por usuario para el train
            self.topics_evaluador_ = self.topics_evaluador_.drop_duplicates(
                    subset=["CVU"],
                    keep="last")

            # merge con evaluadore
            self.topics_evaluador_ = self.df_info_proyectos_
            self.topics_evaluador_ = self.topics_evaluador_[[
                "ID_PROYECTO", "NUMERO_CONVOCATORIA", "ANIO", "CVU"]]

            self.topics_evaluador_ = self.topics_texto_.merge(
                self.topics_evaluador_,
                on=["ID_PROYECTO", "NUMERO_CONVOCATORIA", "ANIO"], how="left")
            # groupby de los vectores por evaluador
            self.topics_evaluador_ = self.topics_evaluador_.groupby(
                ["CVU"]
                )[columnas_numericas[0:]].sum()
        #  descomentr para guardar modelos
            dump(self.topics_evaluador_,
                 './trained_models/archivos/topicos_por_evaluador.pkl')
            dump(self.topics_texto_,
                 './trained_models/archivos/topicos_por_texto_train.pkl')
        else:
            # guardar topicos por texto
            dump(self.topics_texto_,
                 './trained_models/archivos/topicos_por_texto.pkl')
