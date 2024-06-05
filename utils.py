import os
import random
import time
import openai
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone
from langchain_community.llms import Replicate

def chat_inference(query: str, model_choice: str) -> str:
    try:
        os.environ["OPENAI_API_KEY"] = "sensitave data"
        os.environ["PINECONE_API_KEY"] = "sensitave data"
        os.environ["REPLICATE_API_TOKEN"] = "sensitave data"

        pinecone = Pinecone(api_key="sensitave data")
       
        #Prompt
        prompt = PromptTemplate(
            template="You are a helpful assistant {context}",
            input_variables=['context']
        )
        
        # Set up OpenAI Embeddings model
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
                
        # Load Pinecone index and create vector store
        vector_store = PineconeVectorStore(index_name="chatbot", embedding=embedding_model) 
                
        #Similarity Search
        input_documents = vector_store.similarity_search(query, k=2)
                  
        # Determine the chat model based on the user's choice
        if model_choice == "gpt-3.5-turbo":
            chat_llm = ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-3.5-turbo",
                temperature=0,
                verbose=True,
            )
            
            #Chain
            chain = load_qa_chain(
                llm=chat_llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=True
            )
                    
            #Response
            response = chain.run(
                input_documents=input_documents,
                question=query,
            )
            
            return response
        elif model_choice == "gpt-4":
            chat_llm = ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4",
                temperature=0,
                verbose=True,
            )
            
            #Chain
            chain = load_qa_chain(
                llm=chat_llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=True
            )
                    
            #Response
            response = chain.run(
                input_documents=input_documents,
                question=query,
            )
            
            return response
        elif model_choice == "Llama-2-70b-chat":
            chat_llm = Replicate(
                model="meta/llama-2-70b-chat",
                model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
            )
            context = "\n".join(doc.page_content for doc in input_documents)

            # Update the prompt with the context
            formatted_prompt = prompt.format(context=context)
            return chat_llm(formatted_prompt)        
        elif model_choice == "Falcon-40b-instruct":
            chat_llm = Replicate(
                model="joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
                model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
            )
            context = "\n".join(doc.page_content for doc in input_documents)

            # Update the prompt with the context
            formatted_prompt = prompt.format(context=context)
            return chat_llm(formatted_prompt)
        else:
            return "Unknown model selected"   
        
    except Exception as exception:
        error = str(exception)
        print("error: ", error)
        return error

def generate_response(query: str, chat_history: list, model_choice: str) -> tuple:
    response = chat_inference(query, model_choice)
    
    print(response)
    
    chat_history.append((query, response))
    time.sleep(random.randint(0, 5))
    return response, chat_history
