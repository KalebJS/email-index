from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from typing import List, Optional, Tuple

import pandas as pd
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, connections
from retry import retry
from sentence_transformers import SentenceTransformer


@dataclass
class MilvusConnectionSecrets:
    user: str
    password: str
    alias: Optional[str] = "default"
    host: Optional[str] = "localhost"
    port: Optional[str] = "19530"


@dataclass
class QueryResult:
    email_id: int
    score: float


def preload_collection(func):
    @wraps(func)
    def wrapper(self: "MilvusService", *args, **kwargs):
        self.collection.load()
        return func(self, *args, **kwargs)

    return wrapper


class MilvusService:
    index_params = {"metric_type": "COSINE", "index_type": "FLAT"}
    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
    }
    collection_name = "emails"
    index_name = "email_embedding"

    def __init__(
        self,
        credentials: MilvusConnectionSecrets,
        df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        reset: bool = False,
    ):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.connect(credentials)
        self.collection: Collection = self.create_or_get_collection(reset)
        if df is not None:
            self.insert(df)
        self.create_index()
        self.verbose = verbose

    @retry(MilvusException, tries=10, delay=30, logger=None)
    def connect(self, credentials: MilvusConnectionSecrets):
        connections.connect(**credentials.__dict__)

    def create_or_get_collection(self, reset: bool) -> Collection:
        with suppress(Exception):
            collection = Collection(self.collection_name)
            if reset:
                collection.drop()
            else:
                return collection
        id = FieldSchema(name="id", dtype=DataType.INT64, auto_id=True, is_primary=True)
        text_embedding = FieldSchema(name=self.index_name, dtype=DataType.FLOAT_VECTOR, dim=768)
        email_id = FieldSchema(name="email_id", dtype=DataType.INT64)
        schema = CollectionSchema(
            fields=[id, text_embedding, email_id], description="Collection for email embeddings", enable_dynamic=True
        )

        return Collection(name=self.collection_name, schema=schema, using="default", shards_num=2)

    def create_index(self):
        self.collection.create_index(field_name=self.index_name, index_params=self.index_params)

    def insert(self, df: pd.DataFrame):
        df[self.index_name] = self.embedding_model.encode(df["text"].tolist()).tolist()
        df = df[[self.index_name, "email_id"]]
        self.collection.insert(df)

    @preload_collection
    def search(self, query: str, k: Optional[int] = 10, threshold: float = -1 * float("inf")) -> List[QueryResult]:
        vector = self.embedding_model.encode([query])
        query_results = self.collection.search(
            data=vector,
            anns_field=self.index_name,
            param=self.search_params,
            limit=k,
            expr=None,
            output_fields=["email_id", "id"],
        )
        results = []
        for result in query_results[0]:
            item = QueryResult(email_id=result.email_id, score=result.distance)
            if item.score < threshold:
                continue
            results.append(item)

        return results

    def __sizeof__(self) -> int:
        return self.collection.num_entities

    def __len__(self) -> int:
        return self.__sizeof__()
