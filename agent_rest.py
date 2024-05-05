
from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel

from spoke import (
    retrieve_context, load_context_retrieval, load_chroma,
    disease_entity_extractor_v2, relations_similarity_extractor, get_vector_similarity, convert_ssr_to_node_hits, get_full_node_context, get_pruned_node_context
)

vectorstore = load_chroma()
embedding_function_for_context_retrieval = load_context_retrieval()

app = FastAPI()

class ResponseModel(BaseModel):
    exception: bool = True
    context: Optional[str] = None

class ResponseModelInfo(ResponseModel):
    entities: Optional[list[str]] = None
    relations: Optional[str] = None

class ResponseModelDebug(ResponseModelInfo):
    full_context: Optional[list[str]] = None

# Creating an APIRouter
kg_router = APIRouter()

@kg_router.get("/", include_in_schema=False)
def read_root():
    return {
    "This is the REST API for Knowledge Graph based context retrieval. Avalible commands:":[
    "/sample_questions",
    "/get_kg_disease_context/{nl_question}",
    "/get_kg_disease_context_info/{nl_question}",
    "/get_kg_disease_context_debug/{nl_question}"]
    }


@kg_router.get("/sample_questions", description="Returns a list of sample questions from a text file.", include_in_schema=False)
async def sample_questions():
    try:
        return FileResponse("questions.txt")
    except Exception as e:
        raise HTTPException(status_code=404, detail="File not found")

@kg_router.get("/get_kg_disease_context/{question}", response_model=ResponseModel,
               description="Returns KG-based context for the natural-language question about disease.")
async def get_kg_disease_context(question: str):
    try:
        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval)
        if context:
            return {"exception": False, "context": context}
        else:
            return {"exception": False, "context": ""}
    except Exception as e:
        return {"exception": True, "context": "Error"}  # Returning an empty dictionary if any error occurs during context retrieval

def get_kg_disease_context_verbose(question: str, debug: bool=False):
    try:
        entities = disease_entity_extractor_v2(question)
        relations = relations_similarity_extractor(question)
        if entities:
            max_number_of_high_similarity_context_per_node, node_hits = get_vector_similarity(vectorstore, entities)
        else:
            max_number_of_high_similarity_context_per_node, node_hits = convert_ssr_to_node_hits(vectorstore, question)
        full_node_context = [""]
        if debug:
            full_node_context = get_full_node_context(node_hits)
        context = get_pruned_node_context(
            embedding_function_for_context_retrieval,
            max_number_of_high_similarity_context_per_node,
            node_hits,
            relations
        )
        return entities, relations, context, full_node_context
    except Exception as e:
        return "Error", "Error", None, "Error"

@kg_router.get("/get_kg_disease_context_info/{question}", response_model=ResponseModelInfo,
               description="Returns KG-based context and relations for the natural-language question about disease.")
async def get_kg_disease_context_info(question: str):
    try:
        entities, relations, context, full = get_kg_disease_context_verbose(question, False)
        exception = False
        if entities == "Error":
            exception = True
        if context:
            return {"exception": exception, "entities":entities, "relations":relations, "context": context}
        else:
            return {"exception": exception, "entities":entities, "relations":relations, "context": ""}
    except Exception as e:
        return {"exception": True}  # Returning an empty dictionary if any error occurs during context retrieval

@kg_router.get("/get_kg_disease_context_debug/{question}", response_model=ResponseModelDebug,
               description="Returns KG-based context and debug info for the natural-language question about disease.")
async def get_kg_disease_context_debug(question: str):
    try:
        entities, relations, context, full_context = get_kg_disease_context_verbose(question, True)
        exception = False
        if entities == "Error":
            exception = True
        if context:
            return {"exception": exception, "entities":entities, "relations":relations, "context": context, "full_context":full_context}
        else:
            return {"exception": exception, "entities":entities, "relations":relations, "context": "", "full_context":full_context}
    except Exception as e:
        return {"exception": True}  # Returning an empty dictionary if any error occurs during context retrieval

def custom_openapi():
    if kg_router.openapi_schema:
        return kg_router.openapi_schema
    openapi_schema = get_openapi(
        title="Knowledge Graph REST API",
        version="1.0",
        description="This API retrieves context related to disease queries using a SPOKE Knowledge Graph.",
        terms_of_service="https://agingkills.eu/terms/",
        routes=kg_router.routes,
    )
    openapi_schema["servers"] = [
        {"url": "https://kg.agingkills.eu"},
        {"url": "http://localhost:8777"}
    ]
    kg_router.openapi_schema = openapi_schema
    return kg_router.openapi_schema

kg_router.openapi = custom_openapi
app.include_router(kg_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8777)
