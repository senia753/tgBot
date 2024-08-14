import telebot
from telebot import types
import PyPDF2
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Токен бота
bot = telebot.TeleBot("7074991526:AAEAKiG419r4aAaNxHhDYIpfjrDV45NU55A", parse_mode=None)

command = "start"

# Вставка ключа Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyASDLRRhBg-8ySqEzbrzk1uOWQg395Ckd4"

# Загрузка документа
text = ""
file = open("c:/tmp/python/text_interview.pdf", "rb")
pdf = PyPDF2.PdfReader(file)
for page in pdf.pages:
    text += page.extract_text()

# Разбиение документа на части
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len
)
texts = text_splitter.split_text(text)

# Создание векторного хранилища
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_texts(texts, embedding=embeddings)
vector_store.as_retriever()
vector_store.save_local("faiss_local")

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.4, max_length=10000)
system_prompt = """This file contains data on 31 python interview question. Each person has a description of their age, 
    profession, salary, work experience, and additional information. All the questions are numbered. Text written 
    under question applies to that person until the next question.) Answer the question in as much detail as 
    possible, given the context provided. Answer only in Russian\n\nContext:\n {context}?\nQuestion: \n{question}\nAnswer:"""


#@bot.message_handler(commands=['document'])
#def get_document(message):
#    global command
#    command = 'start'
#    print(text)
#    bot.send_message(message.from_user.id, text)

@bot.message_handler(commands=['start'])
def get_start(message):
    global command
    command = 'start'
    text = ('Добро пожаловать! Я готов ответить на Ваши вопросы или просто поболтать. Чего бы Вы хотели сегодня?\nПолезные команды:\n/explore_doc - задать вопросы по содержанию файла'
            '\n/small_talk - пообщаться с ботом на другие темы.\n/start - стартовое сообщение\n/help - вывести все команды бота')
    print(text)
    bot.send_message(message.from_user.id, text)

@bot.message_handler(commands=['help'])
def get_help(message):
    global command
    command = 'help'
    response = ('Полезные команды:\n/explore_doc - задать вопросы по содержанию файла'
            '\n/small_talk - пообщаться с ботом на другие темы.\n/start - стартовое сообщение\n/start - привестсвенное сообщение бота\n/help - вывести все команды бота')
    print(response)
    bot.send_message(message.from_user.id, response)

@bot.message_handler(commands=['explore_doc'])
def get_gemini_response_rag(message):
    global command
    command = 'explore_doc'
    response = ('Можете задать вопрос по содержанию документа\nЧто бы Вы хотели узнать? ')
    print(response)
    bot.send_message(message.from_user.id, response)

@bot.message_handler(commands=['small_talk'])
def get_gemini_response_free_talk(message):
    global command
    command = 'small_talk'
    response = 'Это режим свободной беседы, напишите что-нибудь!'
    print(response)
    bot.send_message(message.from_user.id, response)

@bot.message_handler(content_types=['text', 'sticker'])
def get_gemini_response(message):

    bot.send_chat_action(message.chat.id, 'typing')
    
    if command == 'small_talk':
        response = llm.invoke(message.text)
        print(response)
        bot.send_message(message.chat.id, response.content)
        
        if message.sticker:
            bot.send_message(message.chat.id, 'Вы отправили стикер.')

    if command == 'explore_doc':
    
        db = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)
        info = db.similarity_search(message.text)
        
        prompt = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents": info, "question": message.text}, return_only_outputs=True)

        print(response)
        bot.send_message(message.from_user.id, response['output_text'])

    if command == 'start':
        get_start(message)



bot.polling(none_stop=True, interval=0)
