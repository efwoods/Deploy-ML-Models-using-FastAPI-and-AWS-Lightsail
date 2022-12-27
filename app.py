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
# local dictionary implementation:
#     quotes_dictionary = {
#     "quotes":
#         ["All our knowledge has its origins in our perceptions - Leonardo DaVinci",
#         "All knowledge which ends in words will die as quickly as it came to life, with the exception of the written word: which is its mechanical part - Leonardo DaVinci", 
#         "It had long since come to my attention that people of accomplishment rarely sat back and let things happen to them. They went out and happened to things. - Leonardo DaVinci",
#         "I love those who can smile in trouble, who can gather strength from distress, and grow brave by reflection. 'Tis the business of little minds to shrink, but they whose heart is firm, and whose conscience approves their conduct, will pursue their principles unto death. - Leonardo DaVinci",
#         "Learning never exhausts the mind - Leonardo DaVinci","Don't let your self and your thoughts be the biggest obstacle in your way. - Rana el Kaliouby","Let us touch the dying, the poor, the lonely and the unwanted according to the graces we have received and let us not be ashamed or slow to do the humble work. - Mother Teresa","Anger, pain, resentment, and even hate are best healed through compassionate conversation, not through censorship and derision. - Lex Fridman", "I will not shy away from challenging conversations with folks on the left and right, always seeking understanding through empathy, curiosity, and compassion. Understanding alleviates hate. - Lex Fridman", "History of life on Earth is full of love and suffering. I believe that in the long-run: love wins. - Lex Fridman","Sometimes a kind word from a stranger can make me completely forget whatever concerns were weighing heavy on my heart. Kindness like that is a small gift that can make the biggest difference. I was lucky to get that today. I'll try to pass it on tomorrow. - Lex Fridman","Humanity is facing the threat of nuclear war. Conversation between leaders from a place of strength, empathy, wisdom, and love is the way out. The survival of human civilization depends on it. - Lex Fridman","We live inside a simulation and are ourselves creating progressively more realistic and interesting simulations. Existence is a recursive simulation generator. - Lex Fridman","Resentment and cynicism suffocate the human spirit. Choose optimism, and fight for the best possible future you can imagine. - Lex Fridman","I love to work hard, and to surround myself with people who work hard. We live in a culture where that's sometimes looked down upon. But alas, I am who I am. You be you. I'll be me. Deal with it. 😎 - Lex Fridman", "Practice an awareness of where inefficiencies are, and more and more of those areas where you can go an make an adjustment and make something that may effect millions or billions of people in the world will present themselves. Make it better. - John Carmack","It appears that consciousness is a very rare and precious thing and we should take whatever steps we can to preserve the light of consciousness. - Elon Musk","If you're doing something that has high value to people and frankly even if it's something...if it's just a little game or, you know, the system improvement in photo sharing or something, if it has a small amount of good for a large number of people, I mean I think that's fine. - Elon Musk", "When someone you love becomes a memory, that memory becomes a treasure. - Anonymous","I find it amazing that in the cold black void of space there exists, not one, but billions of stars; each of which with the possibility of hosting life.", "When you can't see the stars, become one.", "Finish what you started - Jeff Woods", "Work hard, Play hard - Jeff Woods", "Is anybody out there?! *Distant Voice* Shut UP! *Jeff again* Thank you! - Jeff Woods","We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard - John F. Kennedy","Not only do we live among the stars, the stars live within us. - Neil De Grasse Tyson","No one is better than anyone else. We are all equal. We all are working towards mastery on something we love. - Chamath Palihapitiya","The reports of my death are greatly exaggerated. - Mark Twain", "The difference between the right word and the almost right word is the difference between lightning and a lightning bug. - Mark Twain","You are a good person. - Cindy Woods","Stay humble. - Cindy Woods","I love you to the stars and beyond! - Cindy Woods","God is like feeling a hand on your heart. - Cindy Woods","Share your knowledge. It is a way to achieve immortality. - Dalai Lama", "As people alive today, we must consider future generations: a clean environment is a human right like any other. It is, therefore, part of our responsibility toward others to ensure that the world we pass on is as healthy, if not healthier than we found it. - Dalai Lama","The goal is not to be better than the other man, but your previous self. - Dalai Lama","Love and compassion are necessities, not luxuries. Without them, humanity cannot survive. - Dalai Lama","The more you are motivated by love, The more fearless & free your actions will be. - Dalai Lama","Cogito, ergo sum. - Renee Descartes","Veritas - Harvard Motto","Mens et Manus - MIT Motto","Time is relative; its only worth depends upon what we do as it is passing. - Albert Einstein","Try not to become a man of success, but rather try to become a man of value. - Albert Einstein","That is the way to learn the most, that when you are doing something with such enjoyment that you don't notice that the time passes. - Albert Einstein","Early to bed and early to rise makes a man healthy, wealthy and wise. - Benjamin Franklin","I will give unto him that is athirst of the fountain of the water of life, freely. - Revalations 21:6","I'm sorry I don't want to be an emperor. That's not my business. I don't want to rule or conquer anyone. I should like to help everyone if possible: Jew, Gentile, black man, white. We all want to help one another. Human beings are like that. - Charlie Chapplain","We want to live by each other's happiness not by each other's misery. We don't want to hate and despise one another in this world. There's room for everyone and the good earth is rich and can provide for everyone. The way of life can be free and beautiful. - Charlie Chapplain","More than machinery, we need humanity. More than cleverness, we need kindness and gentleness. Without these qualities life will be violent and all will be lost. - Charlie Chapplain", "The aeroplane and the radio have brought us closer together the very nature of these inventions cries out for the goodness in men; cries out for universal brotherhood; for the unity of us all. - Charlie Chapplain", "You are not cattle, you are men. You have the love of humanity in your hearts. You don't hate. Only the unloved hate. The unloved and the unnatural. - Charlie Chapplain","'The kingdom of God is within man; not one man or a group of men, but in all men.' In you! You the people have the power ... to create machines ... to create happiness ... to make this life free and beautiful! To make this life a wonderful adventure! - Charlie Chapplain","Why do we fall, Bruce? So we can learn to pick ourselves up. - Thomas Wayne","It is what you do from now on that will either move our civilization forward a few tiny steps, or else... begin to march us steadily backward. - Patrick Stewart","The World awaited Armageddon; instead, something miraculous happened. We began to use atomic energy not as a weapon, but as a nearly limitless source of power. People enjoyed luxuries once thought the realm of science fiction. - Fallout 4","Life and death appeared to me ideal bounds, which I should first break through, and pour a torrent of light into a dark world. - Mary Shelley","Nothing in life is to be feared, it is only to be understood. Now is the time to understand more so that we may fear less. - Mary Shelley","You've gotta take a shot, you have to live at the edge of your capabilities. You gotta live where you're almost certain you're going to fail. Failure actually helps you to recognize the areas where you need to evolve. So fail early, fail often, fail forward. - Will Smith","","Can you fail and still be strong. Can you not fit in and accept yourself. Can you lose everything and still keep searching. Can you be in the dark and still believe in the light. - Kevin Hart","For all the times you supported me with patience and love... Thank you. For all the times I forgot to say it: I love you. - 'Anonymous'", "You become the words you say. - Cindy Woods","Last time I asked: 'What does mathematics mean to you?', and some people answered: 'The manipulation of numbers, the manipulation of structures.' And if I had asked what music means to you, would you have answered 'The manipulation of notes?' - Serge Lang","Scientia Potentia Est - Sir Francis Bacon"]
# }

#     quote_number = random.randint(0, len(quotes_dictionary["quotes"]))
#     fav_quote = json.dumps(quotes_dictionary["quotes"][quote_number], indent = 4)
    
    # return fav_quote["quotes"][quote]
    
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