import httpx
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from jet.memory.memgraph import refresh_auth_token
from jet.logger import logger

paths = [
    "/graph/generate-cypher-queries",
    "/graph/query-graph",
]


class AuthMemgraphRetryOn401Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if the response status code is 401
        response = await call_next(request)

        if any(partial_path in request.url.path for partial_path in paths):
            if response.status_code == 401:
                # Refresh the authentication token
                new_token = refresh_auth_token()

                # Modify the request headers with the new token
                headers = {**dict(request.headers),
                           'Authorization': f'Bearer {new_token}'}

                # Create a new request with the updated headers
                modified_request = Request(
                    scope=request.scope, receive=request.receive)
                modified_request._headers = headers

                # Retry the request with the new Authorization header
                logger.info("Retrying request with updated token.")
                response = await call_next(modified_request)

        return response
