from pipeline.algoritmos_entrenamiento import AlgoritmosPipeline
from pipeline.cleaning_steamming import CleanTools


class TopicModeler():

    def __init__(self):
        self.TPT = None

    def fit(self, df_textos):
        self.TPT = AlgoritmosPipeline(archivo_nmf="blab.pkl",
                                      archivo_tfidf="blob.pkl")
        ct = CleanTools()
        ct.text_cleaner(df=df_textos)
        ct.stem_sentence_apply(limpiar=False)

        self.TPT.guardar_identificadores(df=ct.df_, status="test")
        self.TPT.tfidf_vectorizador_test(texto=ct.df_["DESCRIPCION_PROYECTO"],
                                         tfidf_vectorizer=self.tfidf_vectorizer_)

        self.TPT.producto_punto_topicos(self.matriz_nmf_)

        self.TPT.topicos_a_evaluador()
        topicos_por_texto_ = self.TPT.topic_dot_values_

        return self.topicos_por_texto_

if __name__ == "__main__":
    # https://github.com/jtibshirani/text-embeddings/blob/blog/src/main.py
    topicos_evaluador = pickle.load("a.pkl")
    datos_evaluadores = pickle.load("norberto.pkl")
    for evaluador in topicos_evaluador:
        rcea = topicos_evaluador['CVE_RCEA']
        datos_este_eval = datos_evaluadores[rcea]
        es_document = {'nombre':datos_este_eval['nombre'],
                       'vector_topics':evaluador[2:]}
        ES.create_document(index='evaluadores',es_document)


    propuestas = cargar_las_propuestas()
    df = extraer_textos_y_ids(propuestas)
    tm = TopicModeler()
    vect_topicos_por_text = tm.fit(df)
    for row in vect_topicos_por_text:
        project_id = row['ID_PROYECTO']
        institucion_ = propuestas[project_id]['INST']
        area = propuestas[project_id]['AREA']
        disc = propuestas[project_id]['disc']

        query_es = {'area':area,
                    'disc':disc,
                    'sort_vector': row[0:-1]}
        evals_sugeridos_ordenados = ES.query(query_es)
