import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
import pdf2image
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)
import base64
import hashlib
import pickle
import pandas as pd
from utils import classify_image, classify_pdf
from flask import Flask, request, redirect, session, url_for, render_template

import json, random, requests
from dotenv import dotenv_values

app = FastAPI(
    title="Document Classifier API",
    description="""An API for classifying documents into different categories""",
)

config = dotenv_values('./config/.env')

client_id = config["CLIENT_ID"]
client_secret = config["CLIENT_SECRET"]
auth_url = "https://twitter.com/i/oauth2/authorize"
token_url = "https://api.twitter.com/2/oauth2/token"
redirect_uri = config["REDIRECT_URI"]

# Now we can set the permissions you need for your bot by defining scopes. You can use the authentication mapping guide to determine what scopes you need based on your endpoints. 
scopes = ["tweet.read", "users.read", "tweet.write", "offline.access"]

# Since Twitter’s implementation of OAuth 2.0 is PKCE-compliant, you will need to set a code verifier. This is a secure random string. This code verifier is also used to create the code challenge.
code_verifier = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8")
code_verifier = re.sub("[^a-zA-Z0-9]+", "", code_verifier)

# In addition to a code verifier, you will also need to pass a code challenge. The code challenge is a base64 encoded string of the SHA256 hash of the code verifier.
code_challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
code_challenge = base64.urlsafe_b64encode(code_challenge).decode("utf-8")
code_challenge = code_challenge.replace("=", "")

def make_token():
    return OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scopes)


@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Document Classifier API 📚
    An API classifying documents into different categories
    Note: add "/redoc" to get the complete documentation.
    """
    global twitter
    twitter = make_token()
    authorization_url, state = twitter.authorization_url(
        auth_url, code_challenge=code_challenge, code_challenge_method="S256"
    )
    session["oauth_state"] = state
    return redirect(authorization_url)
    return note


@app.post("/document-classifier")
async def get_document(file: UploadFile = File(...)):
    files = await file.read()
    # save the file
    filename = "filename.pdf"
    with open(filename, "wb+") as f:
        f.write(files)
    # open the file and return the file name
    try:
        data = classify_pdf("filename.pdf")
        return data
    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        return "Unable to parse document! Please upload a valid PDF file."


@app.post("/classify-image")
async def get_image(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        data = classify_image("images.jpg")
        return data
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e

@app.get("/test", response_class=PlainTextResponse, tags=["test"])
async def home():
    note = """
    This is new once.
    """
    return note

@app.get("/tweets", response_class=PlainTextResponse, tags=["tweets"])
async def tweets():
    url = "https://api.twitter.com/2/users/1537504318496047106/tweets?max_results=100"
    prev_quotes = requests.request("GET", url).json()
    tweets = json.dumps(prev_quotes)
    return tweets

    # def get_prior_tweets():
    
@app.get("/quotes", response_class=PlainTextResponse, tags=["quotes"])
async def quotes():

    url = "https://efwoods.github.io/EvanWoodsFavoriteQuotes/quotesTwitterDB.json"
    request_quote = requests.request("GET", url).json()
    quote_num = random.randint(0, len(request_quote["quotes"]))
    new_quote = json.dumps(request_quote["quotes"][quote_num])
    return new_quote


    # try:
    #     quote = parse_fav_quote()
    #     return quote
    # except ValueError as e:
    #     return quote

# def parse_fav_quote():
#     url = "https://efwoods.github.io/EvanWoodsFavoriteQuotes/quotesTwitterDB.json"
#     fav_quote = requests.request("GET", url).json()
#     quote = random.randint(0, len(fav_quote["quotes"]))
#     return fav_quote["quotes"][quote]

def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./Sentiment-LRv1.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

@app.get("/main", response_class=PlainTextResponse, tags=["quotes"])
async def main():
    # Loading the models.
    vectoriser, LRmodel = load_models()
    
    # Text to classify should be in a list.
    text = ["I hate twitter.",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good",
            "I love you 3000",
            "God is like feeling a hand on your heart."]
    
    df = predict(vectoriser, LRmodel, text)
    print(df.head())