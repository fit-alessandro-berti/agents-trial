import os
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

from crewai_tools import tool

@tool
def executor(sql_query: str):
    """
    Executes the provided SQL query against the event data

    Parameters
    ------------------
    sql_query
        SQL query

    Returns
    ------------------
    result
        Result obtained executing the SQL query
    """
    return "0"


from crewai import Crew, Process, Agent, Task

pm_sql_expert = Agent(role="sql expert", goal="An expert in DuckDB SQL.", backstory="Graduated in computer science.")

pm_sql_executor = Agent(role="sql executor", goal="Needs to execute SQL queries on the data.", backstory="Computer scientist.", tools=[executor])

sql_generation = Task(description="Provide one DuckDB SQL statement.", expected_output="One DuckDB SQL statement.", agent=pm_sql_expert)

sql_execution = Task(description="Execute the SQL statement", expected_output="Result of the execution of the SQL statement.", agent=pm_sql_executor)

crew = Crew(agents=[pm_sql_expert, pm_sql_executor], tasks=[sql_generation, sql_execution], verbose="2")
crew.kickoff()
