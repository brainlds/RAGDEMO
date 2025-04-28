import requests
import logging
from datetime import datetime
import csv
from pymilvus import Collection
import sys
import os
from pymilvus import connections
import os
from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import utility

from app.config.pathconfig import BASE_DIR


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "poems")
CHAOJIYING_USERNAME = os.getenv("CHAOJIYING_USERNAME")
CHAOJIYING_PASSWORD = os.getenv("CHAOJIYING_PASSWORD")
CHAOJIYING_SOFT_ID = os.getenv("CHAOJIYING_SOFT_ID")
ACCOUNT_EMAIL = os.getenv("ACCOUNT_EMAIL")
ACCOUNT_PASSWORD = os.getenv("ACCOUNT_PASSWORD")

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.rag.rag import DashScopeEmbedding
from app.crawler.chaojiying import Chaojiying_Client

def crawl_and_save_to_milvus():
    """
    爬取古诗文网站的诗歌并保存到Milvus向量数据库
    
    """
    try:
        url ='https://www.gushiwen.cn/user/login.aspx?from=http://www.gushiwen.cn/user/collect.aspx'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0'
        }
        
        #获取登录页面
        resp = requests.get(url, headers=headers)
        content = resp.text

        
        #获取__VIEWSTATE
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'lxml')
        
        VIEWSTATE = soup.select('#__VIEWSTATE')[0].attrs.get('value')
        
        #获取__VIEWSTATEGENERATOR
        VIEWSTATEGENERATOR = soup.select('#__VIEWSTATEGENERATOR')[0].attrs.get('value')
        
        #获取验证码图片
        img_code = soup.select('#imgCode')[0].attrs.get('src')
        img_url = 'https://www.gushiwen.cn' + img_code
        
        #下载图片，验证码
        session = requests.session()
        resp_code = session.get(img_url)
        code_content = resp_code.content
        
        # 确保 code 目录存在
        code_dir = os.path.join(BASE_DIR, 'code')
        #'wb' 二进制写入到 code 目录
        code_path = os.path.join(code_dir, 'code.jpg')
        with open(code_path, 'wb') as f:
            f.write(code_content)
        
        haojiying = Chaojiying_Client(CHAOJIYING_USERNAME, CHAOJIYING_PASSWORD, CHAOJIYING_SOFT_ID)	#用户中心>>软件ID 生成一个替换 96001
        im = open(code_path, 'rb').read()		
        code_name = haojiying.PostPic(im, 1902)['pic_str']
        logger.info(f"验证码: {code_name}")
           
        

            
        #点击登录
        post_url = 'https://www.gushiwen.cn/user/login.aspx?from=http://www.gushiwen.cn/user/collect.aspx'
        post_data = {
            '__VIEWSTATE': VIEWSTATE,
            '__VIEWSTATEGENERATOR': VIEWSTATEGENERATOR,
            'from': 'http://www.gushiwen.cn/user/collect.aspx',
            'email': ACCOUNT_EMAIL,
            'pwd': ACCOUNT_PASSWORD,
            'code': code_name,
            'denglu': '登录'
        }
        
        resp = session.post(post_url, headers=headers, data=post_data)
        if resp.status_code == 200:
            logger.info('登录成功')
        else:
            logger.error('登录失败')
            return {"status": "error", "message": "登录失败", "data": None}
        
        
        resp_collect = session.get('https://www.gushiwen.cn')
        content = resp_collect.text
        
        with open('login.html', 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 读取HTML文件
        with open('login.html', 'r', encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 找到所有诗歌的容器（每首诗在一个<div class="sons">里）
        poems = []
        for sons_div in soup.find_all('div', class_='sons'):
            title_tag = sons_div.find('p')
            author_tag = sons_div.find('p', class_='source')
            content_tag = sons_div.find('div', class_='contson')
        
            if title_tag and author_tag and content_tag:
                title = title_tag.get_text(strip=True)
                
                author_links = author_tag.find_all('a')
                if len(author_links) >= 2:
                    author = author_links[0].get_text(strip=True)
                    dynasty = author_links[1].get_text(strip=True)
                    author_full = f"{dynasty}·{author}"
                else:
                    author_full = author_tag.get_text(strip=True)
        
                
                content = content_tag.get_text(separator='\n', strip=True)
        
                poems.append({
                    'title': title,
                    'author': author_full,
                    'content': content,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
        # 确保 poems 目录存在
        poems_dir = os.path.join(BASE_DIR, 'poems')
        os.makedirs(poems_dir, exist_ok=True)
        
        filename = os.path.join(poems_dir, 'poems'+ datetime.now().strftime('%Y-%m-%d')+'.csv')
        # 写入CSV
        with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=['title', 'author', 'content', 'created_at'])
            writer.writeheader()  # 写表头
            for poem in poems:
                writer.writerow(poem)
        logging.info(f"数据已保存到{filename}")
        
        logging.info("开始处理并插入数据...")
        # 准备插入数据
        data_entities = []
        
        embedding_model = DashScopeEmbedding()
        
        for poem in poems:
            title = poem['title']
            author = poem['author']
            content = poem['content']
            created_at_str = poem['created_at']
        
            # 将 created_at 字符串转成时间戳（秒）
            created_at_timestamp = int(datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S").timestamp())
        
            # 使用阿里云DashScope获取content的embedding
            try:
                # embed_query用于单个文本，返回单个向量
                embedding = embedding_model.embed_query(content)
                
                # 创建一条完整的实体记录
                entity = {
                    "title": title,
                    "author": author,
                    "content": content,
                    "embedding": embedding,
                    "created_at": created_at_timestamp
                }
                data_entities.append(entity)
                
            except Exception as e:
                logging.error(f"处理{title}向量化时出错: {str(e)}")
        
        # 插入数据到Milvus
        if data_entities:
            # 打印调试信息
            logging.info(f"处理的诗歌数量: {len(data_entities)}")
            logging.info(f"第一首诗标题: {data_entities[0]['title']}")
            logging.info(f"向量维度: {len(data_entities[0]['embedding'])}")
            
            try:
                # 连接到Milvus服务器
                connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
                
                # 检查集合是否存在，不存在则创建
                has_collection = utility.has_collection(MILVUS_COLLECTION_NAME)
                logger.info(f"集合'{MILVUS_COLLECTION_NAME}'是否存在: {has_collection}")
                if not has_collection:
                    
                    fields = [
                        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
                        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=256),
                        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),  # 增加content字段的最大长度
                        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                        FieldSchema(name="created_at", dtype=DataType.INT64)
                    ]
                    
                    # 创建集合模式
                    schema = CollectionSchema(fields)
                    
                    # 创建集合
                    collection = Collection(name=MILVUS_COLLECTION_NAME, schema=schema)
                    
                    # 创建向量索引
                    index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
                    collection.create_index(field_name="embedding", index_params=index_params)
                    logging.info(f"集合'{MILVUS_COLLECTION_NAME}'已创建并建立索引")
                else:
                    collection = Collection(MILVUS_COLLECTION_NAME)
                    logging.info(f"已连接到现有集合'{MILVUS_COLLECTION_NAME}'")
                
                # 准备按字段分离的数据
                titles = [entity["title"] for entity in data_entities]
                authors = [entity["author"] for entity in data_entities]
                contents = [entity["content"] for entity in data_entities]
                embeddings = [entity["embedding"] for entity in data_entities]
                created_at = [entity["created_at"] for entity in data_entities]
                
                # 插入数据
                collection.insert([
                    titles,
                    authors,
                    contents,
                    embeddings,
                    created_at
                ])
                
                collection.flush()
                logging.info(f"共插入{len(data_entities)}条数据")
                return {"status": "success", "message": f"成功爬取并存储{len(data_entities)}首诗", "data": {"count": len(data_entities)}}
            except Exception as e:
                logging.error(f"插入Milvus数据库时出错: {str(e)}")
                return {"status": "error", "message": f"插入数据库出错: {str(e)}", "data": None}
        else:
            logging.info("没有数据插入")
            return {"status": "warning", "message": "没有数据可插入", "data": None}
    except Exception as e:
        logging.error(f"爬虫执行出错: {str(e)}")
        return {"status": "error", "message": f"爬虫执行出错: {str(e)}", "data": None}

# 当直接运行此脚本时执行
# if __name__ == "__main__":
#     result = crawl_and_save_to_milvus()
#     print(f"爬虫执行结果: {result}")

















