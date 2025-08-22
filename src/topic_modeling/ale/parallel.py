import os
import json
from joblib import Parallel, delayed
from prerocessing_with_embed import preprocess_text_nlp_aws
import extract_data as ed
from bertopic import BERTopic
from umap import UMAP
from sklearn.cluster import KMeans
from keybert import KeyBERT
from sklearn.decomposition import PCA
from grouping import grouping
from optimaizer_umap_cons_weight import optimizer
import pandas as pd
import multiprocessing
import boto3
import time






def process_data(category, lang):
    try:
        start_time = time.time()
        
        df= pd.read_csv()
        # Preprocessa il testo
        testo, post = preprocess_text_nlp_aws(df, lang=lang)

        # Se il numero di righe in 'post' è inferiore a 3, usa PCA e KMeans con meno cluster
        if post.shape[0] < 10:
            pca_model = PCA(n_components=min(1, post.shape[1])) 
            km = KMeans(n_clusters=post.shape[0])
            bert = BERTopic(embedding_model=None, hdbscan_model=km, umap_model=pca_model)
        else:
            # Ottimizza i parametri UMAP e KMeans
            opt = optimizer(post, k_min=15, k_max=60)
            umap_model = UMAP(n_neighbors=opt['best_umap_params']['n_neighbors'],
                            min_dist=opt['best_umap_params']['min_dist'])
            km = KMeans(n_clusters=opt['best_k'], random_state=42)
            bert = BERTopic(embedding_model=umap_model, hdbscan_model=km)

        # Esegui il clustering con BERTopic
        topics, probs = bert.fit_transform(testo, post)

        # Ottieni le informazioni sui topics
        output = bert.get_topic_info()

        # Aggiungi i topics al dataframe
        df['topics'] = topics

        # Raggruppa i dati
        group = grouping(df, category_col='topics', cols_to_aggregate=['ID', 'link'])
        group['topics'] = category + '_' + group['topics'].astype(str)
        group['Keywords'] = output['Representation']
        group['Representative_Docs'] = output['Representative_Docs']

        # Esporta come JSON con il nome della categoria
        json_out = group.to_json(orient='records')
        output_filename = f"{category}.json"
        output_path = os.path.join('/Users/alessandro.piccolo/Desktop/Neuralword/production/foraws/output', output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_out)

        print(f"Risultati salvati nel file {output_filename}")
        end_time = time.time()
        execution_time = end_time - start_time

        # Scrivo la categoria e il tempo di esecuzione nel file test.txt
        with open('/Users/alessandro.piccolo/Desktop/Neuralword/production/foraws/test.txt', 'a') as f:
            f.write(f"Categoria: {category}, Tempo di esecuzione: {execution_time} secondi\n")
        # Restituisci il nome del file JSON creato
        return output_filename
    except Exception as e:
        # Scrivi la categoria e l'errore nel file errori.txt
        with open('/Users/alessandro.piccolo/Desktop/Neuralword/production/foraws/errori.txt', 'a') as error_file:
            error_file.write(f"Categoria: {category}, Errore: {str(e)}\n")
        print(f"Errore per la categoria {category}: {str(e)}")
        return None

def main():
    # Carica il file JSON con categorie e lingue
    with open('/Users/alessandro.piccolo/Desktop/Neuralword/production/foraws/category_list.json', 'r') as f:
        category_data = json.load(f)

    # Estrarre le coppie category-lang
    category_list = [(item['Sotto-categoria'], item['Lang']) for item in category_data]

    # Parallelizza l'elaborazione delle categorie, limitando a 4 task alla volta con joblib
    results = Parallel(n_jobs=4)(delayed(process_data)(category, lang) for category, lang in category_list)

    # Stampa i nomi dei file JSON creati
    for result in results:
        if result:
            print(result)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Set the multiprocessing method
    main()
