# my-mcp

## Project Purpose

This project provides a Model Context Protocol (MCP) server implementation, enabling seamless integration and interaction with various tools and documentation resources. It is designed to facilitate access to technical documentation and web content extraction through a unified API.

## Available Tools

### 1. fetch_url_content

- **Description:** Fetches the HTML or text content of a website from a provided URL parameter.
- **Parameters:**
  - `url` (str): URL to fetch (http/https)
- **Example:**
  ```json
  {
    "url": "https://example.com"
  }
  ```

### 2. fetch_doc_snippet

- **Description:** Extracts a relevant snippet from the documentation of a technology based on a question.
- **Parameters:**
  - `question` (str): User question
  - `technology` (str): Technology name (e.g. MCP, LangChain, LangGraph)
- **Example:**
  ```json
  {
    "question": "How to use resources?",
    "technology": "MCP"
  }
  ```

## Available Resources

### 1. [MCP Documentation](https://modelcontextprotocol.io/llms-full.txt)
- **Description:** Complete documentation of the MCP protocol

### 2. [LangGraph Documentation](https://langchain-ai.github.io/langgraph/llms-full.txt)
- **Description:** Complete documentation of LangGraph

### 3. [LangChain Documentation](https://python.langchain.com/llms.txt)
- **Description:** Description of all pages in the LangChain documentation

### 4. [TanStack Query Documentation](https://gitdocs1.s3.amazonaws.com/digests/tanstack-query/2a94f68f-b5e5-4456-8344-dd0b6bc184b8.txt)
- **Description:** Complete documentation of TanStack Query (React Query)

### 5. [TanStack Router Documentation](https://gitdocs1.s3.amazonaws.com/digests/tanstack-router/7f1c7910-67f9-46ae-992b-27b6beef2dfc.txt)
- **Description:** Complete documentation of TanStack Router

### 6. [shadcn Documentation](https://gitdocs1.s3.amazonaws.com/digests/shadcn-ui-ui/1e13259c-ef2f-40ac-8001-cd6d21da5bca.txt)
- **Description:** Complete documentation of shadcn

## Getting Started

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure environment variables by copying `.env.example` to `.env` and updating as needed.
4. Run the server:
   ```
   python server.py
   ```

## Connexion à un client MCP

Pour connecter ce serveur MCP à un client compatible (par exemple Kilocode), il suffit d'ajouter la configuration suivante dans le fichier `.kilocode/mcp.json` de votre projet client :

```json
{
  "mcpServers": {
    "mcp-online": {
      "timeout": 300,
      "type": "streamableHttp",
      "url": "your_url/mcp/",
      "disable": false,
      "autoApprove": [
        "search",
        "fetch_url_content"
      ]
    }
  }
}
```

Cela permet d'accéder aux outils et ressources documentaires exposés par `mcp-online` directement depuis votre environnement de développement.

## Deployment

Use the `deploy.sh` script to automate deployment steps.

## License

MIT License (or specify your license here).
