import sqlalchemy
from collections import defaultdict

from rich.pretty import pretty_repr


def sqllite_db_to_prompt(db_path: str) -> str:
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    inspector = sqlalchemy.inspect(engine)
    schema = defaultdict(list)

    for table_name in inspector.get_table_names():
        for column in inspector.get_columns(table_name):
            schema[table_name].append(column['name'])
    return pretty_repr(dict(schema))