'''
milvus 脚本
进行milvus的操作
'''
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility,db,MilvusClient
from dotenv import load_dotenv
import os
import logging
from typing import List

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

#milvus 连接 
class MilvusInstance:
    def __init__(self, db_name:str, collection_name:str):
        self.db_name = db_name
        self.collection_name = collection_name


    def create_database(self):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        db.create_database(db_name=self.db_name)
        logging.info(f"Database '{self.db_name}' 创建成功")

    def create_collection(self, fields:List[FieldSchema])->Collection:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT,db_name=self.db_name)
        if not utility.has_collection(self.collection_name):
            logging.info(f"Collection '{self.collection_name}' 不存在，正在创建...")

            schema = CollectionSchema(fields, description="Poems collection")
            collection = Collection(name=self.collection_name, schema=schema)
            # 建索引加速搜索
            index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
            collection.create_index(field_name="embedding", index_params=index_params)
            logging.info(f"Collection '{self.collection_name}' 创建成功")
        else:
            collection = Collection(self.collection_name)
            logging.info(f"Collection '{self.collection_name}' 已存在")
        return collection


    #插入数据
    def insert_data(self, data:List[dict]):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(self.collection_name)
        collection.insert(data)
        logging.info(f"数据插入成功")
        collection.flush()

    def drop_collection(self):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        utility.drop_collection(self.collection_name)
        logging.info(f"Collection '{self.collection_name}' 删除成功")






