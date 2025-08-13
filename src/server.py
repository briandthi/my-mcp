"""
MCP Server exposant des ressources documentaires et des outils d’extraction de contenu,
conforme aux standards MCP (2025-06-18), avec métadonnées, validation, gestion d’erreur
et ressource de découverte dynamique pour modèles LLM.
"""

from mcp.server.fastmcp import FastMCP
from pydantic import SecretStr
import requests
import dotenv
import re

dotenv.load_dotenv()

mcp = FastMCP(
    name="Documentation MCP",
    port=9001,
    host="0.0.0.0",
)


def fetch_doc(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print(f"Fetched {len(response.text)} characters from {url}")
        if len(response.text) > 5_000_000:
            return "Error: Resource too large"
        return response.text
    except Exception as e:
        return f"Error fetching documentation: {e}"


@mcp.resource(
    uri="https://modelcontextprotocol.io/llms-full.txt",
    name="MCP Documentation",
    description="Complete documentation of the MCP protocol",
    mime_type="text/plain",
)
def read_mcp_doc() -> str:
    return fetch_doc("https://modelcontextprotocol.io/llms-full.txt")


@mcp.resource(
    uri="https://langchain-ai.github.io/langgraph/llms-full.txt",
    name="LangGraph Documentation",
    description="Complete documentation of LangGraph",
    mime_type="text/plain",
)
def read_langgraph_doc() -> str:
    return fetch_doc("https://langchain-ai.github.io/langgraph/llms-full.txt")


@mcp.resource(
    uri="https://python.langchain.com/llms.txt",
    name="LangChain Documentation",
    description="Description of all pages in the LangChain documentation",
    mime_type="text/plain",
)
def read_langchain_doc() -> str:
    return fetch_doc("https://python.langchain.com/llms.txt")


@mcp.resource(
    uri="https://gitdocs1.s3.amazonaws.com/digests/tanstack-query/2a94f68f-b5e5-4456-8344-dd0b6bc184b8.txt",
    name="TanStack Query Documentation",
    description="Documentation complète de TanStack Query (React Query).",
    mime_type="text/plain",
)
def read_tanstack_query_doc() -> str:
    """
    Retourne la documentation complète de TanStack Query (React Query).
    """
    return fetch_doc(
        "https://gitdocs1.s3.amazonaws.com/digests/tanstack-query/2a94f68f-b5e5-4456-8344-dd0b6bc184b8.txt"
    )


@mcp.resource(
    uri="https://gitdocs1.s3.amazonaws.com/digests/tanstack-router/7f1c7910-67f9-46ae-992b-27b6beef2dfc.txt",
    name="TanStack Router Documentation",
    description="Documentation complète de TanStack Router.",
    mime_type="text/plain",
)
def read_tanstack_router_doc() -> str:
    """
    Retourne la documentation complète de TanStack Router.
    """
    return fetch_doc(
        "https://gitdocs1.s3.amazonaws.com/digests/tanstack-router/7f1c7910-67f9-46ae-992b-27b6beef2dfc.txt"
    )


@mcp.resource(
    uri="https://gitdocs1.s3.amazonaws.com/digests/shadcn-ui-ui/1e13259c-ef2f-40ac-8001-cd6d21da5bca.txt",
    name="shadcn Documentation",
    description="Documentation complète de shadcn.",
    mime_type="text/plain",
)
def read_shadcn_doc() -> str:
    """
    Retourne la documentation complète de shadcn.
    """
    return fetch_doc(
        "https://gitdocs1.s3.amazonaws.com/digests/shadcn-ui-ui/1e13259c-ef2f-40ac-8001-cd6d21da5bca.txt"
    )


@mcp.tool(
    name="fetch_url_content",
    title="Fetch the content of a URL",
    description=(
        "Fetches the HTML or text content of a website from a provided URL parameter. "
        "Useful for dynamically extracting the content of a web page.\n\n"
        "Example:\n"
        "```json\n"
        '{ "url": "https://example.com" }\n'
        "```\n"
        "Returns:\n"
        "```json\n"
        '{ "content": [{"type": "text", "text": "..."}] }\n'
        "```\n"
    ),
    annotations={
        "title": "Fetch URL Content",
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True,
    },
)
async def fetch_url_content(url: str) -> dict:
    """
    Fetches the HTML or text content from a provided URL.

    Args:
        url: The URL of the site to fetch (must start with http:// or https://).
    Returns:
        An MCP dictionary with the site content (key 'content'), or an error (key 'isError': True).
    Example:
        { "url": "https://example.com" }
    """
    if not isinstance(url, str) or not re.match(r"^https?://", url):
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": "Invalid URL: must start with http:// or https://",
                }
            ],
        }
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if len(response.text) > 1_000_000:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "Error: Resource too large"}],
            }
        return {"content": [{"type": "text", "text": response.text}]}
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error fetching URL: {e}"}],
        }


@mcp.tool(
    name="fetch_doc_snippet",
    title="Fetch Documentation Snippet",
    description=(
        "Extracts a relevant snippet from the documentation of a technology based on a question. "
        "The question will be used in similarity search against the documentation.\n\n"
        "Example:\n"
        "```json\n"
        '{ "question": "How to use resources?", "technology": "MCP" }\n'
        "```\n"
        "Returns:\n"
        "```json\n"
        '{ "content": [{"type": "text", "text": "..."}] }\n'
        "```\n"
    ),
    annotations={
        "title": "Fetch Documentation Snippet",
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True,
    },
)
async def fetch_doc_snippet(question: str, technology: str) -> dict:
    """
    Takes a question and a technology, reads the corresponding MCP resource,
    uses RAG (LangChain + ChromaDB) to return the most relevant splits.
    Requires: pip install langchain langchain-community langchain-core chromadb sentence-transformers

    Example:
        { "question": "How to use resources?", "technology": "MCP" }
    """
    import os

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.vectorstores import InMemoryVectorStore
    except ImportError:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The required dependencies are not installed. "
                        "Please run: pip install langchain langchain-community langchain-core chromadb sentence-transformers"
                    ),
                }
            ],
        }

    resources = await mcp.list_resources()
    uri = None
    for res in resources:
        if (
            technology.lower() in res.name.lower()
            or technology.lower() in str(res.uri).lower()
        ):
            uri = res.uri
            break
    if not uri:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Documentation not found for technology '{technology}'.",
                }
            ],
        }

    try:
        doc_resources = await mcp.read_resource(uri)
    except Exception:
        return {
            "isError": True,
            "content": [
                {"type": "text", "text": f"Error reading documentation '{technology}'."}
            ],
        }
    doc_text = ""
    for resource in doc_resources:
        if resource.content:
            doc_text += resource.content
    if not doc_text:
        return {
            "isError": True,
            "content": [
                {"type": "text", "text": f"No documentation found for '{technology}'."}
            ],
        }

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    docs = splitter.create_documents([doc_text])

    try:
        from langchain_openai.embeddings import AzureOpenAIEmbeddings

        embeddings = AzureOpenAIEmbeddings(
            api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2024-03-01-preview",
            azure_deployment="text-embedding-3-large",
            model="text-embedding-3-large",
        )
    except Exception as e:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Error initializing AzureOpenAIEmbeddings: "
                        "Make sure you have installed azure-openai and set the required environment variables. "
                        f"Details: {e}"
                    ),
                }
            ],
        }

    vectordb = InMemoryVectorStore(embeddings)
    vectordb.add_documents(docs)

    try:
        k = 5
        results = vectordb.similarity_search(question, k=k)
        if not results:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": "No relevant snippet found in the documentation.",
                    }
                ],
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n\n---\n\n".join([doc.page_content for doc in results]),
                }
            ]
        }
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error searching for snippets: {e}"}],
        }


@mcp.resource(
    uri="mcp://discovery/list_tools_resources",
    name="List Tools and Resources",
    description=(
        "Dynamically lists all tools and resources exposed by this MCP server, "
        "including their names, descriptions, expected parameters (for tools), URIs (for resources), "
        "and a JSON example for each.\n\n"
        "Example response:\n"
        "```json\n"
        "{\n"
        '  "tools": [\n'
        '    {"name": "fetch_url_content", "description": "...", "parameters": [{"name": "url", "type": "str"}], "example": {"url": "https://example.com"} }\n'
        "  ],\n"
        '  "resources": [\n'
        '    {"uri": "...", "name": "...", "description": "...", "example": {} }\n'
        "  ]\n"
        "}\n"
        "```\n"
    ),
    mime_type="application/json",
)
def list_tools_resources() -> dict:
    """
    Returns a JSON object listing all tools and resources, with their metadata and example usage.
    """
    # Tools
    tools = [
        {
            "name": "fetch_url_content",
            "description": "Fetches the HTML or text content of a website from a provided URL parameter.",
            "parameters": [
                {
                    "name": "url",
                    "type": "str",
                    "description": "URL to fetch (http/https)",
                }
            ],
            "example": {"url": "https://example.com"},
        },
        {
            "name": "fetch_doc_snippet",
            "description": "Extracts a relevant snippet from the documentation of a technology based on a question.",
            "parameters": [
                {"name": "question", "type": "str", "description": "User question"},
                {
                    "name": "technology",
                    "type": "str",
                    "description": "Technology name (e.g. MCP, LangChain, LangGraph)",
                },
            ],
            "example": {"question": "How to use resources?", "technology": "MCP"},
        },
    ]
    # Resources
    resources = [
        {
            "uri": "https://modelcontextprotocol.io/llms-full.txt",
            "name": "MCP Documentation",
            "description": "Complete documentation of the MCP protocol",
            "example": {},
        },
        {
            "uri": "https://langchain-ai.github.io/langgraph/llms-full.txt",
            "name": "LangGraph Documentation",
            "description": "Complete documentation of LangGraph",
            "example": {},
        },
        {
            "uri": "https://python.langchain.com/llms.txt",
            "name": "LangChain Documentation",
            "description": "Description of all pages in the LangChain documentation",
            "example": {},
        },
        {
            "uri": "https://gitdocs1.s3.amazonaws.com/digests/tanstack-query/2a94f68f-b5e5-4456-8344-dd0b6bc184b8.txt",
            "name": "TanStack Query Documentation",
            "description": "Documentation complète de TanStack Query (React Query).",
            "example": {},
        },
        {
            "uri": "https://gitdocs1.s3.amazonaws.com/digests/tanstack-router/7f1c7910-67f9-46ae-992b-27b6beef2dfc.txt",
            "name": "TanStack Router Documentation",
            "description": "Documentation complète de TanStack Router.",
            "example": {},
        },
        {
            "uri": "https://gitdocs1.s3.amazonaws.com/digests/shadcn-ui-ui/1e13259c-ef2f-40ac-8001-cd6d21da5bca.txt",
            "name": "shadcn Documentation",
            "description": "Documentation complète de shadcn.",
            "example": {},
        },
    ]
    return {"tools": tools, "resources": resources}


if __name__ == "__main__":
    # Utilisation d'un port non standard pour le déploiement, écoute sur toutes les interfaces réseau
    mcp.run(transport="streamable-http")
