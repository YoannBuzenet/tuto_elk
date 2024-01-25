import json
from pprint import pprint
import os
import time

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()


class Search:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch('http://localhost:9200',timeout=150)
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def create_index(self):
        print("okay buddy1")
        self.es.indices.delete(index='my_documents', ignore_unavailable=True)
        self.es.indices.create(index='my_documents', mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                }
            }
        })    
        print("okay buddy1.5")


    def insert_document(self, document):
        return self.es.index(index='my_documents', document={
            **document,
            'embedding': self.get_embedding(document['summary']),
        })

    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'my_documents'}})
            operations.append({
                **document,
                'embedding': self.get_embedding(document['summary']),
            })
        print("okay buddy30")
        return self.es.bulk(operations=operations)
    
    def reindex(self):
        self.create_index()
        print("okay buddy2")
        with open('data.json', 'rt') as f:
            documents = json.loads(f.read())
            print("okay buddy3")
        print("okay buddy4")    
        return self.insert_documents(documents)
    
    def search(self, **query_args):
        return self.es.search(index='my_documents', **query_args)
    
    def retrieve_document(self, id):
        return self.es.get(index='my_documents', id=id)
    
    def get_embedding(self, text):
        return self.model.encode(text)
