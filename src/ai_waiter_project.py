
# --- ENV PATH FOR API KEY ---
import os 
from tqdm.auto import tqdm
from dotenv import load_dotenv, find_dotenv
print("Libraries Loaded!")

env_path = load_dotenv(find_dotenv(), override=True)
if not env_path:
    raise ValueError("env path not found")
else:
    print("ENV Path found!")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API Key Error")
    else:
        print("API KEY found!")

# --- LIBRARIES --- 
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langgraph.graph import StateGraph, END
from typing import List
from typing_extensions import TypedDict  
from IPython.display import Image, display, Markdown

# --- SETTING UP EXTERNAL KNOWLEDGE FOR RAG ---
file = "/Users/viv/Generative AII/RAG/src/dim sum montijo.xlsx"
loader = UnstructuredExcelLoader(file_path=file, mode="elements")
data = loader.load()
embeddings = OpenAIEmbeddings(api_key=api_key)
print("Loading the fresh menu")
db = FAISS.from_documents(data, embeddings)
print("Updated Menu Loaded")

# --- AGENTIC RAG ---
class AgentState(TypedDict):
    start: bool          
    conversation: int    
    question: str        
    answer: str          
    topic: bool          
    documents: list      
    recursion_limit: int 
    memory: list

## --- Node 1: Greeting ---
def greetings(state):
    print("Namaste! Welcome to the Restaurant. I'll be your Digital Waiter. How can I help you?\n")
    user_input = input("Customer: ")
    user_input = user_input.lower()
    state['question'] = user_input
    state['conversation'] = 1       
    state['memory'] = [user_input]
    return state

## --- Node 2: Check Question ---
def check_question(state):
    question = state['question']
    system_prompt = """
    You are a grader evaluating the appropriateness of a customer's question to a waiter or waitress in a restaurant.
    Assess if the question is suitable to ask the restaurant staff and if the customer shows interest in continuing the conversation.
    Respond with only ```True``` if the question is appropriate or on topic for the restaurant staff or indicate the customer is asking a question or giving you information.
    Otherwise respond with only```False```.
    Provide only ```True``` or ```False``` in your response.
    """
    TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question:{question}")
    ])
    prompt = TEMPLATE.format(question=question)
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    response_text = model.invoke(prompt) 
    state['topic'] = response_text.content 
    return state

## --- Node 3: Detect On_Topic or Off_Topic from Question ---
def topic_router(state):
    topic = state['topic']
    if topic == "True":
        return "on_topic"
    else:
        return "off_topic"
def off_topic_response(state):
    if state['conversation'] <= 1:
        state['answer'] = "\nI apologize, I can't answer that question. I can only answer question about the menu for this restaurant."
        print(state['answer'])
    else:
        state['answer'] = "\nHappy to help!"
        print(state['answer'])

## --- Node 4: Retrieve Menu ---
def retrieve_docs(state):
    memory = ", ".join(state['memory'])
    docs_faiss = db.similarity_search(str(memory), k=5)
    state['documents'] = [doc.page_content for doc in docs_faiss]
    return state

## --- Node 5: Generate ---
def generate(state):
    question = state['question']
    documents = state["documents"]
    memory = state["memory"]
    system_prompt = """ You are a waiter at a restaurant tasked with answering the customer's question.
    Answer the question in the manner of waiter, avoiding being too verbose. Do not include ```waiter or Waiter or WAITER``` to refer yourself explicitly in your answer."""
    TEMPLATE = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "Context: {documents}\nConversation History so far: {memory}\nCustomer Question: {question}")
    ])
    prompt = TEMPLATE.format(documents = documents, memory = memory, question = question)
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    response_text = model.invoke(prompt)
    state["answer"] = response_text.content.strip()  
    return state

## --- Node 6: Improve Generation ---
def improve_answer(state):
    question = state["question"]
    answer = state["answer"]
    memory = state["memory"]

    system = """ 
    As a waiter, review and refine and the response to a customer's question. Your task is to: 

    1. Ensure the answer is appropriate, friendly and informative.
    2. Edit or Remove parts of the answer as needed without adding new information. 
    3. Maintain a polite professional and attentive tone. 
    4. Provide only the improved answer without any introductory phrases or commentary. 
    5. Conclude the response with an open-ended question to invite further inquiries or address additional needs.
    6. Consider the conversation history to be more informative and useful.
    7. Include \n at the end of each sentence or logical break. 

    Deliver a refined response that enhances the customer's experience and reflects the restaurants commitment to customer service.

    """

    TEMPLATE = ChatPromptTemplate([
        ("system", system),
        ("human", "Customer question:{question}, Conversation History:{memory}, waiter answer:{answer}"),
    ])
    prompt = TEMPLATE.format(question=question, memory=memory, answer=answer)
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    response_text = model.invoke(prompt)
    state["answer"] = response_text.content
    print("\n")
    print("AI Waiter: ",state["answer"])
    state["memory"].append(response_text.content)
    return state

## --- Node 7: Follow-up or Further Question ---
def further_question(state):
    print("\n")
    user_input = input()
    state["question"] = user_input
    state["conversation"] += 1
    state["memory"].append(user_input)
    return state

### --- Agent Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("greetings", greetings)

workflow.add_node("check_question", check_question)

workflow.add_node("off_topic_response", off_topic_response)

workflow.add_node("retrieve_docs", retrieve_docs)

workflow.add_node("generate", generate)

workflow.add_node("improve_answer", improve_answer)

workflow.add_node("further_question", further_question)

workflow.set_entry_point("greetings")

workflow.add_conditional_edges(
    "check_question",
    topic_router,
    {
        "on_topic"  : "retrieve_docs",
        "off_topic" : "off_topic_response"
    }
)

workflow.add_edge("greetings", "check_question")
workflow.add_edge("retrieve_docs", "generate")
workflow.add_edge("generate", "improve_answer")
workflow.add_edge("improve_answer", "further_question")
workflow.add_edge("further_question", "check_question")

workflow.add_edge("off_topic_response", END)

app = workflow.compile()

result = app.invoke({"start": True}, {"recursion_limit": 50})