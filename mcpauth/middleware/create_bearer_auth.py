from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.datastructures import Headers

from ..exceptions import (
    MCPAuthBearerAuthException,
    MCPAuthTokenVerificationException,
    MCPAuthAuthServerException,
    MCPAuthConfigException,
    BearerAuthExceptionCode,
    to_rfc6750_error_code,
)
from ..types import AuthInfo, VerifyAccessTokenFunction, Record
from ..utils import BearerWWWAuthenticateHeader, create_resource_metadata_endpoint
from ..auth.token_verifier import validate_token_claims


class BearerAuthConfig(BaseModel):
    """
    Configuration for the Bearer auth handler.
    """

    issuer: Union[str, Callable[[str], None]]
    """
    The expected issuer of the access token. This should be a valid URL or a validation function.
    """

    audience: Optional[str] = None
    """
    The expected audience of the access token. If not provided, no audience check is performed.
    """

    required_scopes: Optional[List[str]] = None
    """
    An array of required scopes that the access token must have. If not provided, no scope check is
    performed.
    """

    resource: Optional[str] = None
    """
    The identifier of the protected resource. When provided, the handler will use the
    authorization servers configured for this resource to validate the received token.
    It's required when using the handler with a `protectedResources` configuration.
    """

    show_error_details: bool = False
    """
    Whether to show detailed error information in the response. Defaults to False.
    If True, detailed error information will be included in the response body for debugging
    purposes.
    """


def get_bearer_token_from_headers(headers: Headers) -> str:
    """
    Extract the Bearer token from the request headers.

    Args:
      headers: The HTTP request headers.

    Returns:
      The Bearer token.

    Raises:
      MCPAuthBearerAuthException: If the Authorization header is missing or invalid.
    """

    auth_header = headers.get("authorization") or headers.get("Authorization")

    if not auth_header:
        raise MCPAuthBearerAuthException(BearerAuthExceptionCode.MISSING_AUTH_HEADER)

    parts = auth_header.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise MCPAuthBearerAuthException(
            BearerAuthExceptionCode.INVALID_AUTH_HEADER_FORMAT
        )

    token = parts[1]
    if not token:
        raise MCPAuthBearerAuthException(BearerAuthExceptionCode.MISSING_BEARER_TOKEN)

    return token


def _handle_error(
    error: Exception,
    resource: Optional[str] = None,
    show_error_details: bool = False,
    required_scopes: Optional[List[str]] = None,
) -> tuple[int, Dict[str, Any], Dict[str, str]]:
    """
    Handle errors from the Bearer auth process.

    Args:
      error: The exception that was caught.
      resource: The resource identifier for which the auth failed.
      show_error_details: Whether to include detailed error information in the response.
      required_scopes: The scopes required by the endpoint, echoed in the
        `WWW-Authenticate` header for `insufficient_scope` errors (RFC 6750).

    Returns:
      A tuple of (status_code, response_body, headers).
    """
    headers: Dict[str, str] = {}
    www_authenticate_header = BearerWWWAuthenticateHeader()
    www_authenticate_header.set_parameter_if_value_exists(
        "realm", resource or "mcp-auth"
    )

    if isinstance(error, (MCPAuthTokenVerificationException, MCPAuthBearerAuthException)):
        www_authenticate_header.set_parameter_if_value_exists(
            "error", to_rfc6750_error_code(error)
        )
        # Only the static, sanitized message is exposed — never the exception
        # cause, which may contain the raw token or a stack trace.
        if error.message:
            www_authenticate_header.set_parameter_if_value_exists(
                "error_description", error.message
            )

    resource_metadata_endpoint = (
        create_resource_metadata_endpoint(resource) if resource else None
    )

    if isinstance(error, MCPAuthTokenVerificationException):
        if resource_metadata_endpoint:
            www_authenticate_header.set_parameter_if_value_exists(
                "resource_metadata", resource_metadata_endpoint
            )
        headers[www_authenticate_header.header_name] = www_authenticate_header.to_string()
        return 401, error.to_json(show_error_details), headers

    if isinstance(error, MCPAuthBearerAuthException):
        is_insufficient_scope = (
            error.code == BearerAuthExceptionCode.MISSING_REQUIRED_SCOPES
        )
        status_code = 403 if is_insufficient_scope else 401
        if is_insufficient_scope and required_scopes:
            www_authenticate_header.set_parameter_if_value_exists(
                "scope", " ".join(required_scopes)
            )
        if status_code == 401 and resource_metadata_endpoint:
            www_authenticate_header.set_parameter_if_value_exists(
                "resource_metadata", resource_metadata_endpoint
            )
        headers[www_authenticate_header.header_name] = www_authenticate_header.to_string()
        return status_code, error.to_json(show_error_details), headers

    if isinstance(error, (MCPAuthAuthServerException, MCPAuthConfigException)):
        response: Record = {
            "error": "server_error",
            "error_description": "An error occurred with the authorization server.",
        }
        if show_error_details:
            response["cause"] = error.to_json()
        return 500, response, headers

    # Re-raise other errors
    raise error


def create_bearer_auth(
    verify_access_token: VerifyAccessTokenFunction,
    config: BearerAuthConfig,
    context_var: ContextVar[Optional[AuthInfo]],
) -> type[BaseHTTPMiddleware]:
    """
    Creates a middleware function for handling Bearer auth.

    This middleware extracts the Bearer token from the `Authorization` header, verifies it using the
    provided `verify_access_token` function, and checks the issuer, audience, and required scopes.

    :param verify_access_token: A function that takes a Bearer token and returns an `AuthInfo` object.
    :param config: Configuration for the Bearer auth handler.
    :param context_var: Context variable to store the `AuthInfo` object for the current request.
    This allows access to the authenticated user's information in later middleware or route handlers.

    :return: A middleware class that handles Bearer auth.
    """

    if not callable(verify_access_token):
        raise TypeError(
            "`verify_access_token` must be a function that takes a token and returns an `AuthInfo` object."
        )

    if isinstance(config.issuer, str):
        try:
            result = urlparse(config.issuer)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL")
        except ValueError:
            raise TypeError("`issuer` must be a valid URL.")
    elif not callable(config.issuer):
        raise TypeError("`issuer` must be either a string or a callable.")

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        """
        Middleware class that handles Bearer auth.

        This class is used to wrap the request handling process and apply Bearer auth checks.
        """

        async def dispatch(
            self, request: Request, call_next: RequestResponseEndpoint
        ) -> Response:
            """
            Dispatch method that processes the request and applies Bearer auth checks.

            Args:
              request: The HTTP request.
              call_next: The next middleware or route handler to call.

            Returns:
              The HTTP response after processing the request.
            """
            try:
                token = get_bearer_token_from_headers(request.headers)
                auth_info = verify_access_token(token)

                if callable(config.issuer):
                    config.issuer(auth_info.issuer)
                    expected_issuer = None
                else:
                    expected_issuer = config.issuer

                validate_token_claims(
                    auth_info,
                    issuer=expected_issuer,
                    audience=config.audience,
                    required_scopes=config.required_scopes,
                )

                if context_var.get() is not None:
                    logging.warning(
                        "Overwriting existing auth info in context variable."
                    )

                context_var.set(auth_info)

                # Call the next middleware or route handler
                response = await call_next(request)
                return response

            except Exception as error:
                logging.error(f"Error during Bearer auth: {error}")
                status_code, response_body, headers = _handle_error(
                    error,
                    config.resource,
                    config.show_error_details,
                    config.required_scopes,
                )
                return JSONResponse(status_code=status_code, content=response_body, headers=headers)

    return BearerAuthMiddleware
