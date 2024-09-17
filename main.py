from fastapi import FastAPI
from gensim.models import Word2Vec
import string
import requests
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures
from pyngrok import ngrok
import nest_asyncio
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class TextParagraph(BaseModel):
    text_paragraph: str


def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = ' '.join(text.split())
    return text

def extract_keywords_with_embeddings(text, keywords, model):
    text = preprocess_text(text)
    words = text.split()
    extracted_keywords = []

    for word in words:
        if word.lower() in keywords:
            extracted_keywords.append(word)

        if word.lower() in model.wv.key_to_index:
            similar_keywords = model.wv.most_similar(word.lower(), topn=1)
            similar_keyword = similar_keywords[0][0] if similar_keywords else None
            if similar_keyword and similar_keyword.lower() in keywords:
                extracted_keywords.append(similar_keyword)

    text_keywords = text.split()
    for keyword in keywords:
        if ' ' in keyword and keyword in text:
            extracted_keywords.append(keyword)

    return extracted_keywords

def fetch_images(keyword, num_images, api_key, cx):
    url = f'https://www.googleapis.com/customsearch/v1?q={keyword}&num={num_images}&searchType=image&key={api_key}&cx={cx}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching images. Status code: {response.status_code}")
        print(f"Error response: {response.text}")
        return []
    data = response.json()
    image_urls = [item['link'] for item in data.get('items', [])]
    return image_urls

def preprocess_image(image_url, target_size=(224, 224)):
    response = requests.get(image_url)
    try:
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        image_array = np.array(image)
        if image_array.max() <= 1.0:
            return image_array.astype(np.float32)
        else:
            image = image.resize(target_size)
            return np.array(image) / 255.0
    except UnidentifiedImageError as e:
        print(f"Error processing image: {e}")
        return None

def identify_images(images, keyword):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    relevant_images = []
    for image_url in images:
        image_array = preprocess_image(image_url)
        if image_array is not None:
            inputs = processor(text=[keyword], images=image_array, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            if probs.item() > 0.5:
                relevant_images.append(image_url)

    return relevant_images

@app.post("/image-urls/")
def get_image_urls(request: TextParagraph):
    keyword_array = ['binary search tree', 'bst', 'left child ', 'root', 'right child nodes',
                     'parent node', 'leaf node', 'insertion', 'deletion', 'traversal', 'height of a tree',
                     'balanced bst', 'avl tree', 'red-black tree', 'search operation', 'in-order predecessor',
                     'in-order successor', 'level-order traversal', 'height-balanced bst',
                     'binary tree rotation', 'binary tree traversal algorithms', 'binary tree properties',
                     'binary tree implementation', 'binary tree operations', 'binary search tree analysis']

    #fill in the api keys and cx key from google console
    api_key = ''
    cx = ''

    # Preprocess the text paragraph
    text_paragraph = preprocess_text(request.text_paragraph)

    # Create a Word2Vec model
    model = Word2Vec([text_paragraph.split()], min_count=0)

    # Extract keywords using word embeddings
    extracted_keywords = extract_keywords_with_embeddings(text_paragraph, keyword_array, model)

    # Fetch and process images based on the extracted keywords
    keyword_image_urls = {}
    for keyword in set(extracted_keywords):
        images = fetch_images(keyword, 5, api_key, cx)
        relevant_images = identify_images(images, keyword)
        keyword_image_urls[keyword] = relevant_images

    return keyword_image_urls


nest_asyncio.apply()
uvicorn.run(app, host='0.0.0.0', port=8000)