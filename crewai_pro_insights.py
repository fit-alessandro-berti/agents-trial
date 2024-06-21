from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
import pm4py
import os


os.environ["OPENAI_API_KEY"] = "NA"


llm = ChatOpenAI(
    model = "qwen2:7b-instruct-q6_K",
    base_url = "http://137.226.117.70:11434/v1")


log = pm4py.read_xes("tests/input_data/fairness/renting_log_high.xes.gz")

dfg_abstraction = pm4py.llm.abstract_dfg(log)

pm_root_cause_analyst = Agent(role="root_cause_analyst", goal="Perform root cause analysis in process mining.", backstory="Finds all the root causes of the performance issues.", llm=llm)
pm_root_cause_explainer = Agent(role="root_cause_explainer", goal="Explaining process mining insights to non-technical people.", backstory="Very detailed analyst.", llm=llm)
pm_insights_grader = Agent(role="insights_grader", goal="Giving a score to the process mining insights proposed by others.", backstory="Judge.", llm=llm)

root_cause_analysis = Task(description="Find the root causes of the performance issues in the process. Only process and data specific considerations, not general considerations. DFG abstraction: {dfg_abstraction}", expected_output="The chain of thought related to the insight", agent=pm_root_cause_analyst)
root_cause_chain_of_thought = Task(description="Explain the first of the provided insight in detail. Original DFG abstraction: {dfg_abstraction}", expected_output="The chain of thought related to the insight", agent=pm_root_cause_explainer)
root_cause_grader = Task(description="Provide a grade from 1.0 (minimum) to 10.0 (maximum) for all the provided root cause insights. Original DFG abstraction: {dfg_abstraction}", expected_output="A list of insights, each accompanied by a grade.", agent=pm_insights_grader)

crew = Crew(agents=[pm_root_cause_analyst, pm_root_cause_explainer, pm_insights_grader], tasks=[root_cause_analysis, root_cause_chain_of_thought, root_cause_grader], verbose="2")
crew.kickoff({"dfg_abstraction": dfg_abstraction})
