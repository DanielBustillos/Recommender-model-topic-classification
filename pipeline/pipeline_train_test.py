from joblib import dump, load
from pipeline.cleaning_steamming import CleanTools
from pipeline.algoritmos_entrenamiento import AlgoritmosPipeline
from pipeline.score import ScorePipelineCoincidencias
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
#  from pipeline.score import ScorePipeline
import pandas as pd


class PipelineScorerTraining():

    def __init__(self):
        self.topicos_evaluador_ = None
        self.topicos_por_texto_ = None
        self.tfidf_vectorizer_ = None
        self.nmf_matrix_ = None

    def train_pipeline(self, df_texto_train,
                       n_features, max_df, min_df,
                       n_components, max_iter,
                       beta_loss, solver, alpha,
                       l1_ratio, thresh_percentile):
        """
        Pipeline para el conjunto de entrenamiento
        """

        print("Training")

        TPT = AlgoritmosPipeline()
        TPT.guardar_identificadores(df_texto_train, status="train")

        TPT.tfidf_vectorizador_train(max_df, min_df, n_features)

        TPT.train_nmf(n_components=n_components,
                      max_iter=max_iter, solver='mu', alpha=.1, l1_ratio=.5)
        TPT.vocabulario_nmf()
        TPT.filtro_vectores_nmf(thresh_percentile)

        TPT.producto_punto_topicos()
        TPT.topicos_a_evaluador()

        self.topicos_evaluador_ = TPT.topics_evaluador_
        self.tfidf_vectorizer_ = TPT.tfidf_vectorizer_
        self.nmf_matrix_ = TPT.nmf_matrix_
        self.topicos_por_texto_ = TPT.topics_texto_

    def test_pipeline(self, df_texto_test):
        """
        Pipeline para el conjunto de score
        """
        print("Testing")

        ct = CleanTools()
        ct.text_cleaner(df=df_texto_test)
        ct.stem_sentence_apply(limpiar=False)
        #
        TPT_test = AlgoritmosPipeline()
        TPT_test.guardar_identificadores(df=ct.df_, status="test")
        TPT_test.tfidf_vectorizador_test(
             texto=ct.df_["DESCRIPCION_PROYECTO"],
             tfidf_vectorizer=self.tfidf_vectorizer_)

        TPT_test.producto_punto_topicos(self.nmf_matrix_)

        TPT_test.topicos_a_evaluador()

        self.topicos_por_texto_ = TPT_test.topics_texto_

    def scorer(self, df_texto_test, evaluadores_a_sugerir=5):
        """
        Obtiene el score del conjunto de prueba
        """

        print("Scoring")
        SPC = ScorePipelineCoincidencias()
        SPC.cosine_similarity_metric(
            df_evaluador_topico_=self.topicos_evaluador_,
            df_texto_topico_=self.topicos_por_texto_,
            df_proyectos_eval_=df_texto_test,
            evaluadores_a_sugerir=evaluadores_a_sugerir)
        return SPC.metrica_mean


class PipelinePropuestaEvaluador():

    def __init__(self):
        self.topicos_evaluador_ = None
        self.topicos_por_texto_ = None
        self.tfidf_vectorizer_ = None
        self.nmf_matrix_ = None

    def test_pipeline(self, df_texto_test, tfidf_vectorizer, matriz_nmf):
        """
        Info:
        """
        print("Testing")
        ct = CleanTools()
        ct.text_cleaner(df=df_texto_test)
        ct.stem_sentence_apply(limpiar=False)
        TPT_test = AlgoritmosPipeline()
        TPT_test.guardar_identificadores(df=ct.df_, status="produccion")
        TPT_test.tfidf_vectorizador_test(
            texto=ct.df_["DESCRIPCION_PROYECTO"],
            tfidf_vectorizer=tfidf_vectorizer)

        TPT_test.producto_punto_topicos(matriz_nmf)
        TPT_test.topicos_a_evaluador()

        self.topicos_por_texto_ = TPT_test.topics_texto_

    def scorer(self, df_texto_test, topicos_evaluador,
               evaluadores_a_sugerir=5):
        """
        Info:
        """
        SPC = ScorePipelineCoincidencias()
        SPC.cosine_similarity_metric(
            df_evaluador_topico_=topicos_evaluador,
            df_texto_topico_=self.topicos_por_texto_,
            df_proyectos_eval_=df_texto_test,
            evaluadores_a_sugerir=evaluadores_a_sugerir)
        return SPC.metrica_mean
