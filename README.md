# JobSeekerAI

**Overview**
An agentic AI application that uses two APIs Active Jobs DB API and LinkedIn Job Search API to fetch job openings based on a user prompt.
The app is built with LangChain, OpenAI (or LLaMA) for reasoning, and Streamlit for a clean, interactive UI.

**Features**
* Accepts natural language prompts to search for jobs.
* Automatically generates filters for each API (location, keyword, salary, etc.).
* Displays results in a structured, interactive table.
* Includes clickable job links that open in a new tab.
* Fast, agent-driven integration between APIs.

**Setup**
1. Clone the repository
  '''bash
git clone https://github.com/your-username/job-search-agent.git
cd job-search-agent
'''
2. Install dependencies
  '''bash
pip install -r requirements.txt
'''
3. Set up environment variables
  '''
RAPIDAPI_KEY=your_rapidapi_key_here
OPENAI_API_KEY=your_openai_key_here
'''
4. Run the app
'''bash
streamlit run app.py
'''

**Next Steps**
* Add pydantic classes to ensure correct generation of outputs
* Integrate embedding-based search for better matching.
* Prompt tuning to improve the generation of filters by the LLM
* Integrate more APIs
