from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import openai
from config import OPENAI_API_KEY

# import os

# OPENAI_API_KEY = os.environ.get(OPENAI_API_KEY)

def get_response(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Você é um pesquisador e programador assistente"},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

template = """
Você é um expecialista em gerador de Quiz técnico
Crie um quiz com {num_questions} do tipo {quiz_type} sobre o tema: {quiz_context}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(num_questions=5, quiz_type="múltipla escolha", quiz_context="Data Structure") 

chain = LLMChain(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), prompt=prompt)

    