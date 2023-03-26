import json

import openai
import pandas as pd

import guardrails as gd
from utils import get_examples
import sqlite3

from utils import sqllite_db_to_prompt, sqllite_db_to_prompt_tables
from utils import get_examples

rail_spec = """
<rail version="0.1">

<output>
    <sql
        name="generated_sql"
        description="Generate SQL for the given natural language instruction."
        format="bug-free-sql"
        on-fail-bug-free-sql="reask" 
    />
</output>


<prompt>

Here are few text to sql examples:
{{examples}}

Generate a valid SQL query for the following natural language instruction:

{{nl_instruction}}

Here's schema about the database that you can use to generate the SQL query.
Try to avoid using joins if the data can be retrieved from the same table.

{{db_info}}

@complete_json_suffix
</prompt>


</rail>
"""

guard = gd.Guard.from_rail_string(rail_spec)

def nl2sql(nl_instruction: str, db_id: str) -> str:
    db_info = sqllite_db_to_prompt_tables(f'./data/train.db', table=db_id)

    #with open(f'/Users/shreyarajpal/Downloads/spider/database/{db_id}/schema.sql') as f:
    #    db_schema = f.read()
    examples = ['Query: {s0} \n JSON Object: KARSAS"generated_sql": {s1}KARKAS2 \n'.format(s0=el[0], s1=el[1]) for el in get_examples('some', k=10)]
    examples = [e.replace("KARSAS", "{").replace("KARKAS2", "}") for e in examples]

    response = guard(
        openai.Completion.create,
        prompt_params={
            'nl_instruction': nl_instruction,
            'examples': ' '.join(examples),
            'db_info': db_info,
            # 'db_schema': db_schema,
        },
        engine='text-davinci-003',
        temperature=0.0,
        max_tokens=512,
    )
    return response[1]['generated_sql']

def execute(instruction, dbfile = "./data/train.db", db_id = "table_1_1000181_1"):
    db = sqlite3.connect(dbfile)
    query = nl2sql(instruction, db_id = db_id)
    el = pd.read_sql_query(query, db)
    return el 

# print(execute("I need smth"))

