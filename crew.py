import os

# Disable CrewAI telemetry to prevent thread signal errors in Streamlit
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    ScrapeWebsiteTool,
    EXASearchTool
)
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Import custom tools and guardrails
from guardrails import write_report_guardrail
from chart_generator_tool import ChartGeneratorTool


@CrewBase
class ParallelDeepResearchCrew:
    """ParallelDeepResearch crew using Google Gemini LLM"""

    # Declare config file paths as class attributes for @CrewBase
    agents_config = 'config/agents.yaml'  # change to 'config/agents.yaml' if inside a config directory
    tasks_config = 'config/tasks.yaml'    # change to 'config/tasks.yaml' if inside a config directory

    def __init__(self):
        # Fetch GEMINI_API_KEY from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Initialize Gemini model via CrewAI LLM wrapper with retry & backoff safeguards
        self.gemini_llm = LLM(
            model="gemini/gemini-3.1-flash-lite",
            api_key=gemini_api_key,
            max_retries=5,     # Automatically wait and retry on transient/429 errors
            timeout=120        # Allow backoff windows during retries
        )
        self.gemini_llm2 = LLM(
            model="gemini/gemini-3.5-flash-lite",
            api_key=gemini_api_key,
            max_retries=5,     # Automatically wait and retry on transient/429 errors
            timeout=120        # Allow backoff windows during retries
        )

    # Define the agents
    @agent
    def research_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["research_planner"],
            llm=self.gemini_llm,
            verbose=True,
            max_rpm=10
        )

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_researcher"],
            tools=[
                EXASearchTool(base_url=os.getenv("EXA_BASE_URL")),
                ScrapeWebsiteTool()
            ],
            llm=self.gemini_llm2,
            verbose=True,
            max_rpm=10,
            max_iter=15
        )

    @agent
    def fact_checker(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_checker"],
            tools=[
                EXASearchTool(base_url=os.getenv("EXA_BASE_URL")),
                ScrapeWebsiteTool()
            ],
            llm=self.gemini_llm,
            verbose=True,
            max_rpm=10,
            max_iter=15
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],
            llm=self.gemini_llm2,
            tools=[ChartGeneratorTool()],
            verbose=True,
            max_rpm=10,
            max_iter=15
        )

    # Define the tasks
    @task
    def create_research_plan(self) -> Task:
        return Task(
            config=self.tasks_config["create_research_plan"],
        )

    @task
    def research_main_topics(self) -> Task:
        return Task(
            config=self.tasks_config["research_main_topics"],
            async_execution=True,
        )

    @task
    def research_secondary_topics(self) -> Task:
        return Task(
            config=self.tasks_config["research_secondary_topics"],
            async_execution=True,
        )

    @task
    def validate_main_topics(self) -> Task:
        return Task(
            config=self.tasks_config["validate_main_topics"],
        )

    @task
    def validate_secondary_topics(self) -> Task:
        return Task(
            config=self.tasks_config["validate_secondary_topics"],
        )

    @task
    def write_final_report(self) -> Task:
        return Task(
            config=self.tasks_config["write_final_report"],
            guardrails=[write_report_guardrail],
            markdown_output=True,
            output_file="final_report.md"
        )

    # Define the crew
    @crew
    def crew(self) -> Crew:
        """Creates the ParallelDeepResearchCrew crew"""

        # Safely attach knowledge sources only if the file exists
        knowledge_sources = []
        if os.path.exists("user_preference.txt"):
            knowledge_sources.append(
                TextFileKnowledgeSource(file_paths=["user_preference.txt"])
            )

        return Crew(
            agents=self.agents,  # Automatically populated by @agent
            tasks=self.tasks,    # Automatically populated by @task
            memory=False,
            process=Process.sequential,
            tracing=False,
            max_rpm=10,          # Limits total requests across parallel tasks
            verbose=True,
            knowledge_sources=knowledge_sources
        )
