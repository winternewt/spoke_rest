import ast
import json
import os
import numpy as np
import openai
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from joblib import Memory
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential

with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
with open('prompts.yaml', 'r') as f:
    system_prompts = yaml.safe_load(f)

SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
DISEASE_EXTRACTION = system_prompts["DISEASE_ENTITY_EXTRACTION"]
RELATION_EXTRACTION = system_prompts["KG_RELATION_EXTRACTION"]
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
EDGE_EVIDENCE = False

memory = Memory("cachegpt", verbose=0)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)

# Config openai library
config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version

def get_system_prompt(name="KG_RAG_BASED_TEXT_GENERATION"):
    return system_prompts[name]


def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(3))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    #print('Calling OpenAI... p:', system_prompt, '\ni:', instruction)
    response = openai.ChatCompletion.create(
        temperature=temperature,
        deployment_id=chat_deployment_id,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    #print("Debug: Response object:", response)
    if 'choices' in response \
       and isinstance(response['choices'], list) \
       and len(response) >= 0 \
       and 'message' in response['choices'][0] \
       and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'


@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id=None, temperature=TEMPERATURE):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)


def disease_entity_extractor_v2(text):
    chat_model_id = 'gpt-3.5-turbo'
    chat_deployment_id = None
    prompt_updated = DISEASE_EXTRACTION + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, DISEASE_EXTRACTION, chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None

def relations_similarity_extractor(text):
    chat_model_id = 'gpt-3.5-turbo'
    chat_deployment_id = None
    prompt_updated = RELATION_EXTRACTION + "\n" + "Sentence : " + text
    return get_GPT_response(prompt_updated, RELATION_EXTRACTION, chat_model_id, chat_deployment_id, temperature=0)

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)


def load_context_retrieval():
    return load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)

def load_chroma():
    embedding_function = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def get_context_using_spoke_api(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = list(map(lambda x: "pubmedId:"+x, pmid_list))
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:
                        provenance = "SPOKE-KG"
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
  #  merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + "."
  #  context = merge_2.context.str.cat(sep=' ')
  #  context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
    merge_2.loc[:, "context"] = merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + ", source: <" + merge_2.provenance + ">."
    context = merge_2.context.str.cat(sep='\n')
    context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " source: <" + node_context[0]["data"]["properties"]["source"] + ">."
    return context, merge_2

def get_full_node_context(node_hits_list):
    node_context_list = []
    for node_name in node_hits_list:
        context, context_table = get_context_using_spoke_api(node_name)
        node_context_list.append(context)
    return node_context_list

def get_vector_similarity(vectorstore, entity_dict):
    node_hits_list = []
    context_window = int(CONTEXT_VOLUME / len(entity_dict))
    for entity in entity_dict:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits_list.append(node_search_result[0][0].page_content)

    return context_window, node_hits_list

def get_pruned_node_context(embedding_function, context_window, node_hits_list, question):
    question_embedding = embedding_function.embed_query(question)
    node_context_extracted = ""

    for node_name in node_hits_list:
        node_context, context_table = get_context_using_spoke_api(node_name)
        node_context_list = node_context.split(".\n")
        node_context_embeddings = embedding_function.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1),
                                          np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in
                        node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities],
                                             QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD)
        high_similarity_indices = [s[1] for s in similarities if
                                   s[0] > percentile_threshold and s[0] > QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY]
        if len(high_similarity_indices) > context_window:
            high_similarity_indices = high_similarity_indices[:context_window]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        if EDGE_EVIDENCE:
            high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            context_table.loc[:,
            "context"] = context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype(
                'str') + "\n\n"
            node_context_extracted = context_table.context.str.cat(sep='\n')
        else:
            node_context_extracted += ".\n".join(high_similarity_context)
            node_context_extracted += ". "
    return node_context_extracted

def convert_ssr_to_node_hits(vectorstore, question):
    node_hits = vectorstore.similarity_search_with_score(question, k=5)
    context_window = int(CONTEXT_VOLUME / 5)
    node_hits_list = [node[0].page_content for node in node_hits]
    return context_window, node_hits_list

def retrieve_context(question, vectorstore, embedding_function):
    entities = disease_entity_extractor_v2(question)
    relations = relations_similarity_extractor(question)
    if entities:
        context_window, node_hits = get_vector_similarity(vectorstore, entities)
        return get_pruned_node_context(embedding_function, context_window, node_hits, relations)
    else:
        context_window, node_hits = convert_ssr_to_node_hits(vectorstore, question)
        return get_pruned_node_context(embedding_function, context_window, node_hits, question)
