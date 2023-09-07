from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import openai
from config import OPENAI_API_KEY
import streamlit as st


def create_the_quiz_app_template():
    template = """
    Você é um expecialista em gerador de Quiz técnico
    Crie um quiz com {num_questions} do tipo {quiz_type} sobre o tema: {quiz_context}
    O formato de cada pergunta deve ser:
     - múltipla escolha: 
        <pergunta 1>: <a. opção 1>, <b.opção 2>, <c.opção 3>, <d. opção 4>
        <pergunta 2>: <a. opção 1>, <b.opção 2>, <c.opção 3>, <d. opção 4>
        ...
        <resposta 1>: <a|b|c|d>
        <resposta 2>: <a|b|c|d>
        ...
        Exemplo:
        Pergunta 1: Qual a complexidade de tempo do algoritmo de ordenação Bubble Sort?
            a. O(n^2) 
            b. O(n) 
            c. O(nlogn)
            d. O(1)
        Resposta 1: a
    - verdadeiro ou falso:
        <pergunta 1>: <verdadeiro|falso>
        <pergunta 2>: <verdadeiro|falso>
        ...
        <resposta 1>: <verdadeiro|falso>
        <resposta 2>: <verdadeiro|falso>
        Exemplo:
        Pergunta 1: O algoritmo de ordenação Bubble Sort é um algoritmo de ordenação estável?
        Resposta 1: verdadeiro
        
    """

    prompt = PromptTemplate.from_template(template)
    prompt.format(num_questions=5, quiz_type="múltipla escolha", quiz_context="Data Structure")

    return prompt

def create_the_quiz_chain(prompt_template, llm):
    return LLMChain(llm=llm, prompt=prompt_template)

def main():
    st.title("Quiz App")
    st.write("Bem vindo ao Quiz App")
    prompt_template = create_the_quiz_app_template()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chain = create_the_quiz_chain(prompt_template, llm)
    context = st.text_area("Digite o contexto que você quer saber")
    num_questions = st.number_input("Digite o número de questões que você quer", min_value=1, max_value=10)
    quiz_type = st.selectbox("Selecione o tipo de questão", ["múltipla escolha", "verdadeiro ou falso"])
    if st.button("Generete Quizz"):
        quiz_response = chain.run(num_questions=num_questions, quiz_type=quiz_type, quiz_context=context)
        st.write("Quiz gerado com sucesso")
        st.write(quiz_response)

if __name__=="__main__":
    main()
