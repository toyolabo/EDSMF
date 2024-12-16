from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class PoliticalAnalyst():
	"""PoliticalAnalyst crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def news_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config["news_analyst"],
			tools=[FileReadTool()],
			verbose=True,
			allow_delegation=False,
		)

	@agent
	def candidate_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config["candidate_analyst"],
			tools=[FileReadTool()],
			verbose=True,
			allow_delegation=False,
		)

	@agent
	def market_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['market_analyst'],
			verbose=True
		)

	@agent
	def synthesis_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['synthesis_analyst'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def news_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['news_analysis_task'],
			output_file='news_analysis.md'
		)

	@task
	def candidate_policy_task(self) -> Task:
		return Task(
			config=self.tasks_config['candidate_policy_task'],
			output_file='candidate_analysis.md'
		)

	@task
	def market_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['market_analysis_task'],
			output_file='market_analysis.md'
		)

	@task
	def final_synthesis_task(self) -> Task:
		return Task(
			config=self.tasks_config['final_synthesis_task'],
			output_file='final_report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the PoliticalAnalyst crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
