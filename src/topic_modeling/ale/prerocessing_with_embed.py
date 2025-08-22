import pandas as pd
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (normalize_whitespace, remove_punct, remove_eol_characters, remove_stopwords, lower_text, remove_multiple_spaces_and_strip_text, replace_numbers)
from nlpretext.social.preprocess import (remove_mentions, remove_hashtag, remove_emoji)
from nlpretext.token.preprocess import (remove_tokens_with_nonletters, remove_smallwords)
import re
import numpy as np
from tqdm import tqdm
import boto3
import json
tqdm.pandas()

pattern = re.compile('<.*?>')

def nlpretext_it(i,lang='en'):
    preprocessor = Preprocessor()
    #preprocessor.pipe(remove_tokens_with_nonletters)
    preprocessor.pipe(lower_text)
    preprocessor.pipe(remove_emoji)
    preprocessor.pipe(remove_eol_characters)
    preprocessor.pipe(remove_stopwords, args={'lang': lang})
    preprocessor.pipe(remove_punct)
    preprocessor.pipe(normalize_whitespace)
    preprocessor.pipe(remove_multiple_spaces_and_strip_text)# add-on
    #preprocessor.pipe(remove_smallwords, args={'smallwords_threshold' : 2})
    text_clean = preprocessor.run(i)
    return text_clean

def rem_token_lett_combined(excerpt):
    # Split the excerpt into words
    words = excerpt.split(' ')
    # Apply the remove_tokens_with_nonletters function to each word
    processed_words = remove_tokens_with_nonletters(words)
    sentence = ' '.join(processed_words)
    # Rejoin the processed words into a single string
    return sentence

# Initialize the Bedrock Runtime client for AWS

def preprocess_text_nlp_aws(text, columns='excerpt',lang='es',client=None):
    model_id = "amazon.titan-embed-text-v2:0"
    pattern = re.compile('<.*?>')
    
    # Preprocess the text (assuming you have an NLP preprocessing function 'nlpretext_it')
    text[columns] = text[columns].progress_apply(lambda row: re.sub(pattern, '', row))
    text[columns + '_trs'] = text[columns].progress_apply(lambda row: nlpretext_it(row,lang=lang))
    text[columns + '_trs'] = text[columns + '_trs'].progress_apply(lambda row: rem_token_lett_combined(row))
    # Convert the processed text to a list
    testo = text[columns + '_trs'].tolist()
    
    # Prepare to store embeddings
    text_emb = []
    
    # Loop over each text entry in the dataset
    for input_text in tqdm(testo):
        native_request = {"inputText": input_text}
        request = json.dumps(native_request)
        
        # Invoke the model with the request to get embeddings
        response = client.invoke_model(modelId=model_id, body=request)
        
        # Decode the model's native response body
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]
        
        # Store the embedding for this input
        text_emb.append(embedding)
    
    # Convert the list of embeddings to a numpy array
    text_emb = np.array(text_emb)
    
    return testo, text_emb

