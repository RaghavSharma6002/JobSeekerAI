import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import Ollama

# Load API keys
load_dotenv()
RAPIDAPI_KEY = 'b80d2713ffmsh8aa096119256b3cp16f8fajsnfca762cb9ec1'

#Login to HuggingFace
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# API URLs
ACTIVE_JOBS_URL = "https://active-jobs-db.p.rapidapi.com/api/job/getJobs"
LINKEDIN_JOBS_URL = "https://linkedin-job-search-api.p.rapidapi.com/job-search"

# --- API Functions ---
def fetch_active_jobs(query):
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "active-jobs-db.p.rapidapi.com"}
    params = {"query": query, "num_pages": "1"}
    response = requests.get(ACTIVE_JOBS_URL, headers=headers, params=params)
    if response.status_code == 200:
        jobs = response.json().get("data", [])
        return [{"source": "ActiveJobsDB", "title": j.get("title"), "company": j.get("company"),
                 "location": j.get("location"), "link": j.get("url")} for j in jobs]
    return [{"error": f"Active Jobs API Error {response.status_code}"}]

def fetch_linkedin_jobs(query):
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com"}
    params = {"keywords": query, "location": "remote", "page": "1"}
    response = requests.get(LINKEDIN_JOBS_URL, headers=headers, params=params)
    if response.status_code == 200:
        jobs = response.json().get("data", [])
        return [{"source": "LinkedIn", "title": j.get("title"), "company": j.get("company"),
                 "location": j.get("location"), "link": j.get("job_url")} for j in jobs]
    return [{"error": f"LinkedIn API Error {response.status_code}"}]

# --- LangChain Agent ---
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400
)
llm = HuggingFacePipeline(pipeline=pipe)

tools = [
    Tool(
        name="Active Jobs DB Search",
        func=fetch_active_jobs,
        description="Search jobs in Active Jobs DB using job title or skills."
    ),
    Tool(
        name="LinkedIn Jobs Search",
        func=fetch_linkedin_jobs,
        description="Search jobs on LinkedIn using job title or skills."
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# --- Streamlit Interface ---
st.set_page_config(page_title="AI Job Search", page_icon="💼")

st.title("💼 AI-Powered Job Search")
st.markdown("Search real job openings using **Active Jobs DB** and **LinkedIn APIs**")

query = st.text_input("Describe the job you're looking for:", placeholder="e.g. Remote Python Data Scientist in USA")

if st.button("Search Jobs"):
    if not query.strip():
        st.warning("Please enter a job description first.")
    else:
        with st.spinner("Fetching job listings..."):
            try:
                results = agent.run(query)
                st.subheader("Job Results")
                st.write(results)
            except Exception as e:
                st.error(f"Error: {str(e)}")
