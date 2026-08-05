import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool, EXASearchTool
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Custom imports
from guardrails import write_report_guardrail
from chart_generator_tool import ChartGeneratorTool

@CrewBase
class ParallelDeepResearchCrew:
    """ParallelDeepResearch crew using Gemini API"""

    def __init__(self):
        # Global Gemini LLM instance for all agents
        self.gemini_llm = LLM(
            model="gemini/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )

    @agent
    def research_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["research_planner"],
            llm=self.gemini_llm,
            verbose=True
        )

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_researcher"],
            tools=[
                EXASearchTool(base_url=os.getenv("EXA_BASE_URL")),
                ScrapeWebsiteTool()
            ],
            llm=self.gemini_llm,
            verbose=True,
            max_rpm=150,
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
            max_rpm=150,
            max_iter=15
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],
            llm=self.gemini_llm,
            tools=[ChartGeneratorTool()],
            verbose=True,
            max_rpm=150,
            max_iter=15
        )

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

    @crew
    def crew(self) -> Crew:
        """Creates the ParallelDeepResearchCrew crew"""
        # Safely include knowledge source only if file exists
        knowledge_sources = []
        if os.path.exists("user_preference.txt"):
            knowledge_sources.append(
                TextFileKnowledgeSource(file_paths=["user_preference.txt"])
            )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            memory=True,
            process=Process.sequential,
            tracing=False,
            verbose=True,
            knowledge_sources=knowledge_sources
        )
