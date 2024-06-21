from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
import pm4py
import duckdb
import os
from crewai_tools import tool
import pickle


global_variables = {}

os.environ["OPENAI_API_KEY"] = "sk-"

llm = ChatOpenAI(
    model = "gpt-4o",
    base_url = "https://api.openai.com/v1")


@tool("Prompt containing the description of the attributes of event log.")
def get_prompt_base_description_of_event_log() -> str:
    """
    Returns the prompt that can be used to get the base description of the attributes of the event log.

    Returns
    --------------
    base_description_prompt
        Base description prompt
    """
    pickle.dump(global_variables, open("global_variables.dump", "wb"))

    prompt = "Given an event log describing a process with the following directly-follows graph:\n\n"
    prompt += pm4py.llm.abstract_dfg(global_variables['dataframe'], max_len=5000, response_header=False)
    prompt += "\n\nand the following attributes:\n\n"
    prompt += pm4py.llm.abstract_log_attributes(global_variables['dataframe'], max_len=5000)
    prompt += "\n\nCould you make an hypothesis and provide me a DuckDB SQL query that I can execute to filter the dataframe on the cases in which unfairness is likely to happen? It can be an OR query (several different conditions could signal unfairness) The dataframe is called 'dataframe'. Please only a single query. The single query should only look at the case attributes, not the activities in a case. Quote the names of the attributes with a \" at start and at the end."
    prompt += "\nThe SQL query has to be contained between the tags ```sql and ```"
    return prompt


@tool("Executes the provided SQL query.")
def execute_sql_query(sql_query: str):
    """
    Executes the given SQL query.

    Parameters
    ----------------
    sql_query
        SQL query
    """
    pickle.dump(global_variables, open("global_variables.dump", "wb"))

    F = open("sql.txt", "w")
    F.write(str(sql_query))
    F.close()

    sql_query = sql_query.split("```sql")[-1].split("```")[0]
    dataframe = global_variables['dataframe']
    dataframe_prot = duckdb.sql(sql_query).to_df()
    cases = dataframe_prot["case:concept:name"].unique()
    dataframe_nonprot = dataframe[~dataframe["case:concept:name"].isin(cases)]
    global_variables['dataframe_prot'] = dataframe_prot
    global_variables['dataframe_nonprot'] = dataframe_nonprot


@tool("Prompt for process comparison between two groups.")
def get_prompt_for_process_comparison() -> str:
    """
    Gets the prompt that should be used for process comparison.

    Returns
    --------------
    process_comparison_prompt
        Process comparison prompt
    """
    pickle.dump(global_variables, open("global_variables.dump", "wb"))

    dataframe_prot = global_variables['dataframe_prot']
    dataframe_nonprot = global_variables['dataframe_nonprot']
    prompt = "I want to identify the unfair differences between the treatment of the 'protected' group (first) and the 'unprotected' group (second). I report the process variants. Each process variant is also reported with its execution time."
    prompt += "\n\nProcess variants of the protected group:"
    prompt += pm4py.llm.abstract_variants(dataframe_prot, max_len=5000, response_header=False)
    prompt += "\n\nProcess variants of the unprotected group:"
    prompt += pm4py.llm.abstract_variants(dataframe_nonprot, max_len=5000, response_header=False)
    prompt += "\n\nwhich are the main differences? use your domain knowledge."
    return prompt


global_variables['dataframe'] = pm4py.read_xes("tests/input_data/fairness/renting_log_high.xes.gz")
#global_variables['dataframe'] = global_variables['dataframe'][[x for x in global_variables['dataframe'].columns if 'protected' not in x]]

pm_protected_group_identification = Agent(role="protected_group_identification", goal="Identifying the protected group.", backstory="Fairness expert.", llm=llm)
pm_comparison_expert = Agent(role="comparison_expert", goal="Compare the behavior of two groups of cases.", backstory="Comparison expert.", llm=llm)

protected_group_identification = Task(description="Retrieve a SQL query to identify the protected cases and executes that against the event log.", expected_output="No output", agent=pm_protected_group_identification, tools=[get_prompt_base_description_of_event_log, execute_sql_query])

comparison_protected_nonprotected = Task(description="Comparing the behavior of the protected group against the nonprotected group.", expected_output="A list of discriminations", agent=pm_comparison_expert, tools=[get_prompt_for_process_comparison])

crew = Crew(agents=[pm_protected_group_identification, pm_comparison_expert], tasks=[protected_group_identification, comparison_protected_nonprotected], verbose="2")
crew.kickoff()
