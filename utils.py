import sqlalchemy
from collections import defaultdict
import numpy as np
import deeplake
import openai

from lib.dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines

from rich.pretty import pretty_repr

def L2_search(
    query_embedding: np.ndarray, data_vectors: np.ndarray, k: int = 4
) -> list:
    """naive L2 search for nearest neighbors"""
    # Calculate the L2 distance between the query_vector and all data_vectors
    distances = np.linalg.norm(data_vectors - query_embedding, axis=1)

    # Sort the distances and return the indices of the k nearest vectors
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices.tolist()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

ds = deeplake.load('hub://guardsql/wikisql')
ds.checkout("44ebd41f1461dddbe2e1f142dbc6cc245eaaafd7")
embeddings = ds.embeddings[:10].numpy()
dbfile = "./data/test.db"

engine = DBEngine(dbfile)

def get_examples(query: str, k: int = 10):
    query_emb = np.array(get_embedding(query))
    indices = L2_search(query_emb, embeddings, k=k)
    ds_view = ds[indices]

    examples = [ 
        (el["text"].data()["value"], el["sql"].data()["value"])
        for el in ds_view
    ]
    return examples


def sqllite_db_to_prompt(db_path: str) -> str:
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    inspector = sqlalchemy.inspect(engine)
    schema = defaultdict(list)

    for table_name in inspector.get_table_names():
        for column in inspector.get_columns(table_name):
            schema[table_name].append(column['name'])
    return pretty_repr(dict(schema))

def sqllite_db_to_prompt_tables(db_path: str, table="") -> str:
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    inspector = sqlalchemy.inspect(engine)
    schema = defaultdict(list)
    
    for table_name in inspector.get_table_names():
        if table_name != table:
            continue
        for column in inspector.get_columns(table_name):
            schema[table_name].append(column['name'])
            
    return pretty_repr(schema)

if __name__ == '__main__':
    query = "get all iamges"
    examples = get_examples(query)

    qg = ""
    for example in examples:
        print(example[0])
        print(example[1])
        qg = example[1]
        
    gold = engine.execute_query("table_10875694_11", qg, lower=True)

    #prompt = sqllite_db_to_prompt(dbfile)
    #print(prompt)
