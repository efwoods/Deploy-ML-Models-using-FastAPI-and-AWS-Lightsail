import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

from utils import classify_image, classify_pdf

import json, random, requests

app = FastAPI(
    title="Document Classifier API",
    description="""An API for classifying documents into different categories""",
)



@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Document Classifier API 📚
    An API classifying documents into different categories
    Note: add "/redoc" to get the complete documentation.
    """
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