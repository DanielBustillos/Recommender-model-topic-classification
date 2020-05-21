from joblib import dump, load
from pipeline.cleaning_steamming import CleanTools
from pipeline.algoritmos_entrenamiento import AlgoritmosPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.score import ScorePipeline
from pipeline.score import ScorePipelineCoincidencias
from pipeline.pipeline_train_test import PipelineScorerTraining
import pandas as pd
import numpy as np


class PipelinePropuestaEvaluador():

    def __init__(self):
        self.topicos_por_texto_ = None
        self.numero_sugeridos = None
        self.id_proyecto_index = None
        self.evaluadores_entrenado = None

    def production_pipeline(self, df_texto_test,
                            tfidf_vectorizer, matriz_nmf):
        """
        Genera el vector de topicos a textos de convocatorias nuevas usando el
        vectorizador, la matriz NMF y el vector de topicos por evaluador
        generados en el entrenamiento.
        """

        print("Cleaning")
        ct = CleanTools()
        ct.text_cleaner(df=df_texto_test)
        ct.stem_sentence_apply(limpiar=False)
        print(ct.df_.shape)

        TPT_test = AlgoritmosPipeline()
        TPT_test.guardar_identificadores(df=ct.df_, status="produccion")
        TPT_test.tfidf_vectorizador_test(
            texto=ct.df_["DESCRIPCION_PROYECTO"],
            tfidf_vectorizer=tfidf_vectorizer)

        TPT_test.producto_punto_topicos(matriz_nmf)

        TPT_test.topicos_a_evaluador()

        self.topicos_por_texto_ = TPT_test.topic_dot_values_

    def similitud_evaluadores(self, topicos_evaluador, numero_sugeridos=5):
        """
        Encuentra los N evaluadores mas similares usando cosine similarity
        """
        self.numero_sugeridos = numero_sugeridos
        print("Similitud_Evaluadores")
        #  topicos evaluador
        topicos_evaluador.reset_index(drop=False, inplace=True)
        self.evaluadores_entrenado = topicos_evaluador["CVU"].tolist()
        #  topicos propuesta
        self.topicos_por_texto_.drop_duplicates("ID_PROYECTO", inplace=True)
        self.id_proyecto_index = self.topicos_por_texto_[
                                                        "ID_PROYECTO"].tolist()

    def similitud_cosine_similarity(self):
        cos_simi = cosine_similarity(
            topicos_evaluador.iloc[:, 1:],
            self.topicos_por_texto_.iloc[:, :-1])

        df_cos_simi = pd.DataFrame(cos_simi)
        # generamos un DF
        df_evaluadores_recomendados = pd.DataFrame(index=[i for i in range(
            len(self.id_proyecto_index * self.numero_sugeridos))], columns=[
                                                    "id_proyecto",
                                                    "CVU",
                                                    "score_tm"])

        cvu_evaluadores_lista = []
        score_evaluadores_lista = []
        id_proyecto_lista = []

        for proyecto in range(df_cos_simi.shape[1]):
            list_each_proyecto = df_cos_simi.iloc[:, proyecto].tolist()
            top_5_idx = np.argsort(
                list_each_proyecto)[-self.numero_sugeridos:][::-1].tolist()
            [cvu_evaluadores_lista.append(self.evaluadores_entrenado[pos])
                for pos in top_5_idx]

            #  normalized_score = normalize([df_cos_simi.iloc[pos, proyecto]
            # for pos in top_5_idx])
            [score_evaluadores_lista.append(df_cos_simi.iloc[pos, proyecto])
                for pos in top_5_idx]
            [id_proyecto_lista.append(self.id_proyecto_index[proyecto])
                for i in range(self.numero_sugeridos)]

    #  guardamos los resultados en el DF
        df_evaluadores_recomendados["id_proyecto"] = id_proyecto_lista
        df_evaluadores_recomendados["CVU"] = cvu_evaluadores_lista
        df_evaluadores_recomendados["score_tm"] = score_evaluadores_lista

        df_evaluadores_recomendados.to_csv(
            "./trained_models/resultados/propuesta_evaluadores.csv")


if __name__ == "__main__":
    #  archivo con los proyectos a asignar evaluador: (reemplazar nombre
    # del archivo)
    df_proyectos = load(
        "./data/solicitudes_2019/data_evaluar_pn_2019.pkl").sample(100)
    # modelos
    tfidf_vectorizer = load("./trained_models/modelos/tfidf_vectorizer.pkl")
    matriz_nmf = load("./trained_models/modelos/matriz_nmf.pkl")
    # topicos por evaluador
    topicos_evaluador = load(
        "./trained_models/archivos/topicos_por_evaluador.pkl")

    PPE = PipelinePropuestaEvaluador()
    PPE.production_pipeline(df_proyectos, tfidf_vectorizer, matriz_nmf)
    PPE.similitud_evaluadores(topicos_evaluador, numero_sugeridos=5)
    PPE.similitud_cosine_similarity()
