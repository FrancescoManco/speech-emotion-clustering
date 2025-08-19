# speech-emotion-clustering
Speech Italian Emotion Analysis with Clustering and LLM Cluster interpretation

## Descrizione Moduli Implementati

### `src/core/main.py`
Pipeline principale che coordina l'intera analisi. Prende un file audio in input ed esegue automaticamente segmentazione, estrazione feature e post-processing. Produce un singolo file CSV finale con tutte le informazioni necessarie per le analisi successive. Include barra di progresso per monitorare l'avanzamento.

### `src/segmentation/segmentation.py`
Modulo per la segmentazione automatica di file audio. Usa WhisperX per trascrivere, allineare e identificare i parlanti (speaker diarization). Divide l'audio in segmenti individuali e crea file audio separati per ogni segmento. Produce un CSV con informazioni su ogni segmento: speaker, timing, trascrizione e percorso del file audio.

### `src/SpeechCueLLM/extract_audio_feature.py`
Estrae caratteristiche acustiche da ogni segmento audio usando Praat. Calcola features come durata, intensità media, variazione di volume, pitch medio, deviazione standard del pitch, range del pitch, velocità di articolazione e rapporto armoniche-rumore. Produce un CSV con tutte le features numeriche per ogni segmento.

### `src/SpeechCueLLM/postprocess_audio_feature.py`
Converte le features numeriche in categorie interpretabili (basso, medio, alto) e genera descrizioni testuali delle caratteristiche vocali. Crea anche "impressions" che descrivono come potrebbe essere percepito il parlante. Produce il CSV finale con features categorizzate, descrizioni e impressions pronto per l'analisi.

### `src/SpeechCueLLM/syllable_nuclei.py`
Utility per calcolare la velocità di articolazione usando algoritmi Praat. Conta automaticamente le sillabe nell'audio e calcola il rapporto sillabe/tempo per determinare la velocità del parlato. Viene usato internamente dal modulo di estrazione features.

## Note per lo Sviluppo Futuro

- La pipeline è pronta per l'integrazione del modulo di clustering (TODO nel main.py)
- È predisposta per l'aggiunta dell'analisi LLM per interpretazione semantica (TODO nel main.py)
- Tutti i moduli hanno funzioni core separate dai wrapper CLI per facilità di integrazione
