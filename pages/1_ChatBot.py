import utils
import streamlit as st
from streaming import StreamHandler
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title('Chatbot')
st.write('Allows users to interact with the LLM')

#Based on https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py
class Chatbot:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4-0125-preview"

    def setup_chain(self):
        # Initialize Vector Store
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        retriever = vectorstore.as_retriever()

        # Initialize LLM
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)

        # Initialize Conversation memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Initialize Conversation Chain
        chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    obj = Chatbot()
    obj.main()