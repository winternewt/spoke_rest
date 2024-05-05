from spoke import (get_GPT_response, disease_entity_extractor_v2, relations_similarity_extractor,
                   load_context_retrieval, load_chroma, get_vector_similarity, convert_ssr_to_node_hits,
                   get_system_prompt, get_full_node_context, get_pruned_node_context, retrieve_context)

CHAT_MODEL_ID = "gpt-3.5-turbo"
INTERACTIVE = False
embedding_function_for_context_retrieval = load_context_retrieval()
vectorstore = load_chroma()

def interactive(question, system_prompt, llm_type):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_v2(question)
    relations = relations_similarity_extractor(question)
    print("Formulated relations from the prompt:", relations)
    print(" ")
    if entities:
        print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
        print(" ")
        input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
        print("Finding vector similarity ...")
        max_number_of_high_similarity_context_per_node, node_hits = get_vector_similarity(vectorstore, entities)
        print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
        print(" ")
    else:
        print("Extracted no Disease entities from the prompt")
        max_number_of_high_similarity_context_per_node, node_hits = convert_ssr_to_node_hits(vectorstore, question)
        print("Matched entities from SPOKE based on direct similarity search = '{}'".format(", ".join(node_hits)))
        print(" ")

    # input("Press enter for Step 3a - Context extraction from SPOKE")
    # node_context = get_full_node_context(node_hits)
    # print("Extracted Context is : ")
    # print(". ".join(node_context))
    # print(" ")

    input("Press enter for Step 3 - Pruned context extraction")
    node_context_extracted = get_pruned_node_context(
        embedding_function_for_context_retrieval,
        max_number_of_high_similarity_context_per_node,
        node_hits,
        relations
    )
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")

    input("Press enter for Step 5 - LLM prompting")
    enriched_prompt = "Context: " + node_context_extracted + "\n" + "Question: " + question
    output = get_GPT_response(enriched_prompt, system_prompt, llm_type)
    print(output, '\n')


def main():
    print(" ")
    question = input("Enter your question : ")
    if not INTERACTIVE:
        print("Retrieving context from SPOKE graph...")
        context = retrieve_context(question, vectorstore, embedding_function_for_context_retrieval)
        print("Here is the KG-RAG based answer:")
        print("")
        enriched_prompt = "Context: " + context + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, get_system_prompt(), CHAT_MODEL_ID)
        print(output, '\n')
    else:
        interactive(question, get_system_prompt(), CHAT_MODEL_ID)


if __name__ == "__main__":
    main()