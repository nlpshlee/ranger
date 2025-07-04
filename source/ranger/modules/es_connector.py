from _init import *

from ranger.modules.common_const import *
from ranger.modules import common_util

from elasticsearch import Elasticsearch, helpers

###########################################################################################

class ESConnector:
    def __init__(self, url="http://127.0.0.1", port=9200, log_option=LOG_OPTION.STDOUT, conf_dict: dict=None):
        self.log_option = log_option
        self.conf_dict = conf_dict
        
        if conf_dict is None:
            self.host = url + ":" + str(port)
        else:
            self.make_host(conf_dict)
        
        # ElasticSearch 커넥션 생성
        try:
            self.es = Elasticsearch(self.host)
            self.es.info() # print()로 출력해야 확인 가능하고, 연결되었는지 테스트하기 위한 용도
            common_util.logging(f"ESConnector() host : {self.host}, connected : True\n", log_option)
        except:
            self.es = None
            common_util.logging(f"ESConnector() host : {self.host}, connected : False\n", log_option)
    
    '''
        conf(json) dict로 전달 받은 경우, host 설정
            - 'ES'에서만 사용되는 형식이므로 해당 클래스에서만 하드 코딩
              (공용화 하기 애매...)
    '''
    def make_host(self, conf_dict):
        url, port = "", ""

        if "es_connect" in conf_dict:
            es_connect_dict = conf_dict["es_connect"]
            
            if "url" in es_connect_dict:
                url = es_connect_dict["url"]
            if "port" in es_connect_dict:
                port = es_connect_dict["port"]
        
        self.host = url + ":" + port

    ###########################################################################################
    
    def is_available(self):
        if self.es is not None:
            return True
        else:
            return False

    '''
        'index' 확인
        
            - 이미 존재하면 'True'
            - 존재하지 않고, 'mappings'가 주어졌다면 새로 생성하고 'True'
    '''
    def check_index(self, index, mappings=None):
        if self.is_available():
            if self.es.indices.exists(index=index):
                return True
            else:
                try:
                    if mappings is not None:
                        self.es.indices.create(index=index, mappings=mappings)
                        return True
                except:
                    return False
        
        return False

    def delete_index(self, index):
        if self.is_available():
            try:
                self.es.indices.delete(index=index)
            except:
                pass

    # index_st --> index search term (Ex. "tart*")
    def get_cnt_index(self, index_st: str):
        if self.is_available():
            #return len(self.es.indices.get(index=index_st))
            return len(self.es.indices.get_alias(index=index_st))

        return -1
    
    def get_cnt_doc(self, index):
        if self.check_index(index):
            return self.es.count(index=index)["count"]

        return -1
    
    def get_mappings(self, index):
        try:
            mappings = self.conf_dict["indices"][index]["mappings"]
            return mappings
        except:
            return None

    def get_docs(self, index):
        try:
            docs = self.conf_dict["indices"][index]["docs"]
            return docs
        except:
            return None

    ###########################################################################################
    
    '''
        ElasticSearch 검색 함수 (Scroll API 사용)
        
            - index     : 인덱스 명 (Ex. tart_ma_dic_josa)
              fields    : 검색하려는 필드 list (모든 필드를 검색하려면, 'None')
              size      : 문서 검색 크기 (모든 문서를 검색하려면, '-1')
              query     : 검색 조건
              scroll    : 스크롤 API 세션 유지 시간 ('10s'이면 10초간 유지)
    '''
    def search_scroll(self, index, fields: list=None, size=-1, query={"match_all":{}}, scroll="1m"):
        result = []
        
        # 인덱스 자체에 문제가 있으면 'None' 리턴
        if not self.check_index(index):
            return None
        
        if size < 1 or 10000 < size:
            size = 10000

        searched = self.es.search(index=index, size=size, query=query, sort="_doc", scroll=scroll)
        
        # 최초 search()를 실행한 결과 처리
        self.search_scroll_unit(searched, fields, result)
        scroll_id = ""
        
        # 스크롤 API로 다음 search() 결과 처리
        while 1:
            scroll_id_prev = scroll_id
            scroll_id = searched["_scroll_id"]
            
            # 'scroll_id'가 변경되는 경우, 이전 세션은 종료
            if len(scroll_id_prev) != 0 and scroll_id_prev != scroll_id:
                self.es.clear_scroll(scroll_id=scroll_id_prev)
            
            # 다음 스크롤 가져오기
            searched = self.es.scroll(scroll_id=scroll_id, scroll=scroll)
            if len(searched["hits"]["hits"]) <= 0:
                break
            
            self.search_scroll_unit(searched, fields, result)
        
        # 마지막 세션 종료
        if len(scroll_id) > 0:
            self.es.clear_scroll(scroll_id=scroll_id)

        return result

    '''
        ElasticSearch 검색 함수 (스크롤 단위 반복)
    '''
    def search_scroll_unit(self, searched, fields, result: list):
        for doc_temp in searched["hits"]["hits"]:
            doc_source = doc_temp["_source"]
            
            doc = {}
            if fields is None:
                doc = doc_source
            else:
                for field in fields:
                    doc[field] = doc_source[field]

            # 결과 객체에 저장
            if result is not None:
                result.append(doc)

###########################################################################################

    def insert_bulk(self, bulk_datas, batch_size: int=100000):
        try:
            bulk_len, start, end = len(bulk_datas), 0, batch_size

            while 1:
                if bulk_len <= end:
                    helpers.bulk(self.es, bulk_datas[start:])
                    common_util.logging(f"ESConnector.insert_bulk() insert complet : {bulk_len}")
                    break
                
                helpers.bulk(self.es, bulk_datas[start:end])
                common_util.logging(f"ESConnector.insert_bulk() insert complet : {end}")
                
                start = end
                end += batch_size
            return True

        except Exception as e:
            common_util.logging_error("ESConnector.insert_bulk()", e)
            return False

###########################################################################################

    def update(self, index, id, doc):
        try:
            self.es.update(index=index, id=id, doc=doc)
            return True

        except Exception as e:
            common_util.logging_error("ESConnector.update()", e)
            return False

###########################################################################################

''' def test_connector_init():
    es_connector = ESConnector("http://127.0.0.1", 9201)    # --> connected : False
    es_connector = ESConnector("http://127.0.0.1/", 9200)   # --> connected : False
    
    conf_file_path = "C:/nlpshlee/tart/dev/resources/ma/conf/ma_es_conf.json"
    conf_dict = json_util.load_file_to_dict(conf_file_path)
    es_connector = ESConnector(conf_dict=conf_dict)

def test_check_index(es_connector: ESConnector, index):
    # 없으면 --> False
    es_connector.delete_index(index)
    print(es_connector.check_index(index))
    
    conf_file_path = "C:/nlpshlee/tart/dev/resources/ma/conf/ma_es_conf.json"
    conf = json_util.load_file_to_dict(conf_file_path, "utf-8")
    print(f"\nconf_type : {type(conf)}\n\n### conf :\n{conf}")
    
    mappings = conf["indices"][index]["mappings"]
    print(f"\n### mappings :\n{mappings}\n")
    
    # mappings를 넘겨주면, 새로 생성한다.
    print(es_connector.check_index(index, mappings))   # --> True
    print(es_connector.check_index(index))             # --> True

def test_get_cnt(es_connector: ESConnector, index, index_st):
    print(f"get_cnt_index({index_st}) : {es_connector.get_cnt_index(index_st)}\n")
    print(f"get_cnt_doc({index}) : {es_connector.get_cnt_doc(index)}") '''

''' def test_search(es_connector: ESConnector):
    #result = es_connector.search_scroll("tart_ma_dic_josa", ["lex", "tag"], size=2)
    #result = es_connector.search_scroll("tart_ma_dic_josa", ["lex", "tag"])
    #result = es_connector.search_scroll("tart_ma_dic_josa")
    
    #fields = ["category","tag","lex","date_reg","date_mod"]
    #result = es_connector.search_scroll("tart_ma_dic_josa", fields)
    result = es_connector.search_scroll("tart_ma_dic_josa")
    
    file_path = "C:/nlpshlee/tart/dev/resources/ma/dics/josa/josa_list/download.json"
    
    file = file_util.open_file(file_path, mode='w')
    file.write(json_util.to_str(result))
    file.close()
    
    docs = json_util.load_file_to_dict(file_path)
    print(f"len : {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"[{i}] doc ({type(doc)}) : {doc}")
    
    file = file_util.open_file(file_path, mode='w')
    file.write(json_util.to_str(docs))
    file.close() '''

''' def test_update(es_connector: ESConnector):
    index = "tart_monit_apply_dics"
    id = "750443276e930942aafa61697dd35df7f5f44a93"
    doc = {"run_time":"2"}
    es_connector.update(index, id, doc) '''

###########################################################################################

if __name__ == "__main__":
    from ranger.modules import file_util, json_util
    #test_connector_init()
    
    es_connector = ESConnector()
    #test_check_index(es_connector, "tart_ma_dic_josa")
    #test_get_cnt(es_connector, "tart_ma_dic_josa", "tart*")
    #test_search(es_connector)
    #test_update(es_connector)