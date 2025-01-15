import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv()

repo_id="mistralai/Mistral-7B-Instruct-v0.3"

st.set_page_config(page_title="Text Summarization",page_icon="🦜")
st.title("Summarize Text From YouTube or Website")
st.subheader('Summarize URL')

with st.sidebar:
    hf_api_key=st.text_input("Please enter your Groq API key",type="password")
    
if not hf_api_key.strip():
    st.warning("🚨 Please enter your Groq API key in the sidebar to proceed.")
else:
    generic_url = st.text_input("URL", label_visibility="collapsed")
    
    
prompt=""" 
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt_template=PromptTemplate(input_variables=['text'],template=prompt)

if st.button("Summarize the content"):
    
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the correct information")
        
    elif not validators.url(generic_url):
        st.error("The provided URL is incorrect")
        
    else:
        try:
            with st.spinner("Waiting..."):
                llm=HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=150,temperature=0.7,huggingfacehub_api_token=hf_api_key)
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],continue_on_failure=False,ssl_verify=False,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs=loader.load()
                
                summarize_chain=load_summarize_chain(llm=llm,chain_type='stuff',prompt=prompt_template)
                
                summary=summarize_chain.invoke(docs)
                
                st.success(summary['output_text'])
            
        except Exception as e:
            st.exception(f"Exception{e}")
    