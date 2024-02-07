import os, json, base64, whisper, torch
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from rest_framework.views import APIView
from rest_framework.response import Response
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "Insira sua apikey aqui"

 # Carregamento de dados e armazenamento de vetores
   
llm = ChatOpenAI(model_name="Insira o modelo da OpenAI") # Ou altere para um LLM local de preferência, ex: LLama2
loader = DirectoryLoader("Insira o caminho")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings() # Ou altere para um local, ex: all-MiniLM-L6-v2
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Criação do histórido do chat

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


qa_system_prompt = """Responda às perguntas com base apenas no contexto fornecido. Se não souber a resposta, diga que não possui a informação, sem elaborar.

{context}
Responda apenas se houver informação da pergunta nos documentos carregados. Não forneça respostas além do que está nos textos.
Responda exatamente como está nos textos carregados.
Responda apenas em Português do Brasil (PT-BR)."""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

# Cadeia final

rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever 
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)
chat_history = []

# Texto para audio (TTS)

def text_to_audio(result):
    client = OpenAI()

    response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",  # alloy / echo / fable / onyx / nova / shimmer
    input=result
    )

    response.stream_to_file("result.mp3")

    audio_file_path = 'result.mp3'

    with open(audio_file_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

    return audio_data

# View principal (Rota padrão)

class assistant(APIView):

    def post(self, request):
        data = json.loads(request.body)
        question = data.get("query")
        use_audio = data.get("use_audio") 
        result = rag_chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=question), AIMessage(content=result)])

        if use_audio:
            audio_data = text_to_audio(result)
            return Response({"response_text": result, "response_audio": audio_data})
        else:
            return Response({"response_text": result})

# View para transcrição do audio

class transcribe(APIView):
    
    def post(self, request):
        data = json.loads(request.body)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print()
        print('Using device:', device)
        print()

        
        def record():
            audio_record = data.get("audio_record")
            audio = base64.b64decode(audio_record.split(',')[1])
            file_name = 'request_audio.wav'
            with open(file_name, 'wb') as f:
                f.write(audio)
            return file_name
        record_file = record()
        model = whisper.load_model("small")
        result_whisper = model.transcribe(record_file, fp16=False, language= "pt")

        # Informações adicionais
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        transcription = result_whisper["text"]
        return Response({"response_whisper": transcription})