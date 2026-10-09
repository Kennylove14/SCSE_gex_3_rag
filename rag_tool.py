### COMPLETE THE CODE  ###


from rag import get_relevant_context


class KnowledgeBaseSearchTool:
    name = "search_university_it_policy"
    description = (
        "Use this tool to look up official university IT policy information. "
        "Always use this tool before answering user questions about IT services."
    )

    def run(self, user_query: str) -> str:
        return get_relevant_context(user_query)
