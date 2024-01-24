import json
from pprint import pprint
import os
import time

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()


class Search:
    def __init__(self):
        self.es = Elasticsearch('http://localhost:9200',timeout=150)
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def create_index(self):
        print("okay buddy1")
        self.es.indices.delete(index='my_documents', ignore_unavailable=True)
        self.es.indices.create(index='my_documents')  
        print("okay buddy1.5")


    def insert_document(self, document):
        return self.es.index(index='my_documents', body=document)    
    
    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'my_documents'}})
            operations.append(document)
            print("okay buddy20")
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