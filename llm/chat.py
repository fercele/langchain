from langchain.chat_models import ChatOpenAI


def get_conversation_chain(model_option, temperature, streaming=True, callback_handler = StreamingStdOutCallbackHandler()) -> ConversationalRetrievalChain:
    
    llm_condense_question = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    llm_streaming_qa = ChatOpenAI(temperature=temperature, model=model_option, streaming=streaming, callbacks=[callback_handler])

    question_generator = LLMChain(llm=llm_condense_question, prompt=prompts.MY_CONDENSE_QUESTION_PROMPT)
    
    doc_chain = load_qa_chain(llm_streaming_qa, chain_type="stuff", prompt=prompts.MY_QA_PROMPT)

    qa = ConversationalRetrievalChain(
        retriever=get_redis_parent_retriever(), 
        combine_docs_chain=doc_chain, 
        question_generator=question_generator,
        return_source_documents=True,
        rephrase_question=False      #Fernando - este faz com que a pergunta original seja enviada ao LLM, não a rephrased
    )

    return qa

