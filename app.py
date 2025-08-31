import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import traceback
import json
import pandas as pd

# Load API keys
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# API URLs
ACTIVE_JOBS_URL = "https://active-jobs-db.p.rapidapi.com/active-ats-7d"
LINKEDIN_JOBS_URL = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-7d"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Conversational model
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# --- API Functions ---
def fetch_active_jobs(filters: dict):
    """
    Fetch jobs from Active Jobs DB API
    Filters:
        title_filter (str)          - Keywords for the job search
        location (str)       - Job location (city, country, or remote)
        employment_types (str) - Type: fulltime, parttime, contract, internship
        date_posted (str)    - postedToday, last3days, last7days, last30days
        num_pages (str)      - Number of pages to fetch (1-3)
    """
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "active-jobs-db.p.rapidapi.com"
    }
    if isinstance(filters, str):
      filters = json.loads(filters)
    params = {
        "title_filter": filters.get("title_filter", ""),
        "location_filter": filters.get("location_filter", ""),
        "remote": filters.get("remote", ""),
        "ai_experience_level_filter": filters.get("ai_experience_level_filter", "")

    }
    response = requests.get(ACTIVE_JOBS_URL, headers=headers, params=params)
    if response.status_code == 200:
        jobs = response.json()
        return [
            {
                "source": "ActiveJobsDB",
                "title": j.get("title"),
                "company": j.get("company"),
                "location": j.get("location"),
                "link": j.get("url")
            }
            for j in jobs
        ]
    return [{"error": f"Active Jobs API Error {response.status_code}"}]

def fetch_linkedin_jobs(filters: dict):
    """
    Fetch jobs from LinkedIn Job Search API
    Filters:
        keywords (str)        - Keywords for the job search
        location (str)        - Location (city, country, or 'remote')
        experienceLevel (str) - Entry, Mid, Senior, Director, Internship
        datePosted (str)      - past24Hours, pastWeek, pastMonth
        remote (str)          - true/false
    """
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com"
    }
    if isinstance(filters, str):
      filters = json.loads(filters)
    params = {
        "title_filter": filters.get("keywords", ""),
        "location_filter": filters.get("location", ""),
        "seniority_filter": filters.get("seniority_filter", ""),
        "remote": str(filters.get("remote", "true")).lower(),
        "ai_experience_level_filter": filters.get("ai_experience_level_filter", "")
    }
    response = requests.get(LINKEDIN_JOBS_URL, headers=headers, params=params)
    if response.status_code == 200:
        jobs = response.json()
        return [
            {
                "source": "LinkedIn",
                "title": j.get("title"),
                "company": j.get("company"),
                "location": j.get("location"),
                "link": j.get("job_url")
            }
            for j in jobs
        ]
    return [{"error": f"LinkedIn API Error {response.status_code}"}]

# --- LangChain Agent ---

tools = [
    Tool(
        name="Active Jobs DB Search",
        func=fetch_active_jobs,
        description=(
            "Use this tool to search for jobs using the Active Jobs DB API. "
            "You MUST pass a dictionary with these keys. Values of all keys are optional. You can keep them blank.:\n"
            "- title_filter: Job title or keywords (e.g., 'Python Developer')\n"
            "- location_filter : City, country, or 'remote'\n"
            "- remote: true or false\n"
            "- ai_experience_level_filter: Experience range. Acceptable values: 0-2 or 2-5 or 5-10 or 10+\n"
            'Example: {"title_filter": "AI Engineer", "location_filter": "Remote", "remote": "true","ai_experience_level_filter": ""}'
        )
    ),
    Tool(
        name="LinkedIn Jobs Search",
        func=fetch_linkedin_jobs,
        description=(
            "Use this tool to search for jobs on LinkedIn. "
            "You MUST pass a dictionary with these keys. Values of all keys are optional. You can keep them blank.:\n"
            "- title_filter: Job title or keywords (e.g., 'Data Scientist')\n"
            "- location_filter : City, country, or 'remote'\n"
            "- seniority_filter : Entry, Mid, Senior, Director, Internship\n"\
            "- remote : true or false\n"
            "- ai_experience_level_filter: Experience range. Acceptable values: 0-2 or 2-5 or 5-10 or 10+\n"
            'Example: {"title_filter": "Machine Learning Engineer", "location_filter": "India", "seniority_filter": "Senior", "remote": "true","ai_experience_level_filter":""}'
        )
    )
]

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer should be returned as a list of dictionaries/rows. The keys to be included are Title, Company, Location, Link. Example: [{{"Title":"","Company":"","Location":"","Link":""}}]

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

#agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# --- Streamlit Interface ---
st.set_page_config(page_title="AI Job Search", page_icon="💼")

st.title("💼 AI-Powered Job Search")
st.markdown("Just describe the job you are looking for and let AI do the rest!!")

query = st.text_input("Describe the job you're looking for:", placeholder="e.g. Remote Python Data Scientist in USA")

if st.button("Search Jobs"):
    if not query.strip():
        st.warning("Please enter a job description first.")
    else:
        with st.spinner("Fetching job listings..."):
            try:
                results = agent_executor.invoke({"input": query})
                df = pd.DataFrame(json.loads(results['output']))
                df["Link"] = df["Link"].apply(lambda x: f'<a href="{x}" target="_blank">Apply Here</a>')
                st.subheader("Job Results")
                st.write(
                  df.to_html(escape=False, index=False), 
                  unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.text(traceback.format_exc())
