import os
print(os.getcwd())

import asyncio
from pprint import pprint
from routes.rag import QueryRequest, get_nodes


async def main():
    # Create a QueryRequest object with the specified parameters
    query_request = QueryRequest(
        query="Tell me about yourself.",
        chunk_size=1024,
        chunk_overlap=40,
        sub_chunk_sizes=[512, 256, 128],
        with_heirarchy=True,
        top_k=20,
    )

    # Call the get_nodes endpoint
    response = await get_nodes(query_request)

    # Print the response
    pprint(response.dict())

if __name__ == "__main__":
    asyncio.run(main())
