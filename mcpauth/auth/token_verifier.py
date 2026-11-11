from typing import List, Optional

import jwt
from ..config import AuthServerConfig
from ..exceptions import (
    AuthServerExceptionCode,
    BearerAuthExceptionCode,
    MCPAuthAuthServerException,
    MCPAuthBearerAuthException,
    MCPAuthBearerAuthExceptionDetails,
    MCPAuthTokenVerificationException,
    MCPAuthTokenVerificationExceptionCode,
)
from ..types import AuthInfo, VerifyAccessTokenFunction
from ..utils import create_verify_jwt


def validate_token_claims(
    auth_info: AuthInfo,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    required_scopes: Optional[List[str]] = None,
) -> None:
    """
    Shared validation of the claims extracted from an access token. Used by the
    Bearer auth middleware for both the resource server and authorization server
    handlers, so the checks live in exactly one place.

    Each check is only enabled when the corresponding expectation is configured;
    anything left unset stays disabled for backward compatibility.

    :param auth_info: The authentication information extracted from the token.
    :param issuer: The expected issuer. Compared with an exact string match —
        no normalization (e.g. no trailing-slash tolerance) is applied.
    :param audience: The expected audience. The token's `aud` claim may be a
        string or a list of strings; the check passes when the expected audience
        intersects the token's audiences.
    :param required_scopes: Scopes the token must have been granted. All of them
        must be present. An empty list means "no scopes required" and disables
        the check — it is neither "deny all" nor "any scope will do".
    :raises MCPAuthBearerAuthException: If any enabled check fails.
    """
    if issuer is not None and auth_info.issuer != issuer:
        raise MCPAuthBearerAuthException(
            BearerAuthExceptionCode.INVALID_ISSUER,
            cause=MCPAuthBearerAuthExceptionDetails(
                expected=issuer, actual=auth_info.issuer
            ),
        )

    if audience is not None:
        token_audiences = (
            [auth_info.audience]
            if isinstance(auth_info.audience, str)
            else auth_info.audience if isinstance(auth_info.audience, list) else []
        )
        if audience not in token_audiences:
            raise MCPAuthBearerAuthException(
                BearerAuthExceptionCode.INVALID_AUDIENCE,
                cause=MCPAuthBearerAuthExceptionDetails(
                    expected=audience, actual=auth_info.audience
                ),
            )

    if required_scopes:
        granted_scopes = set(auth_info.scopes)
        missing_scopes = [
            scope for scope in required_scopes if scope not in granted_scopes
        ]
        if missing_scopes:
            raise MCPAuthBearerAuthException(
                BearerAuthExceptionCode.MISSING_REQUIRED_SCOPES,
                cause=MCPAuthBearerAuthExceptionDetails(missing_scopes=missing_scopes),
            )


class TokenVerifier:
    """
    Encapsulates all authentication logic and policies for a specific protected resource
    or a legacy `server` configuration.
    """

    def __init__(self, auth_servers: List[AuthServerConfig]):
        self._auth_servers = auth_servers
        self._issuers = {server.metadata.issuer for server in auth_servers}

    def _get_unverified_jwt_issuer(self, token: str) -> str:
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
        except jwt.exceptions.DecodeError as e:
            raise MCPAuthTokenVerificationException(
                MCPAuthTokenVerificationExceptionCode.INVALID_TOKEN,
                cause={"error_description": "The JWT is malformed or invalid.", "cause": e},
            )

        issuer = payload.get("iss")
        if not issuer or not isinstance(issuer, str):
            raise MCPAuthTokenVerificationException(
                MCPAuthTokenVerificationExceptionCode.INVALID_TOKEN,
                cause={
                    "error_description": "The JWT payload does not contain the `iss` field."
                },
            )
        return issuer

    def _get_auth_server_by_issuer(self, issuer: str) -> AuthServerConfig:
        for server in self._auth_servers:
            if server.metadata.issuer == issuer:
                return server
        raise MCPAuthBearerAuthException(
            BearerAuthExceptionCode.INVALID_ISSUER,
        )

    def validate_jwt_issuer(self, issuer: str):
        if issuer not in self._issuers:
            raise MCPAuthBearerAuthException(
                BearerAuthExceptionCode.INVALID_ISSUER,
                cause=MCPAuthBearerAuthExceptionDetails(
                    expected=", ".join(self._issuers), actual=issuer
                ),
            )

    def create_verify_jwt_function(
        self, leeway: float = 60
    ) -> VerifyAccessTokenFunction:
        def verify_jwt(token: str) -> AuthInfo:
            unverified_issuer = self._get_unverified_jwt_issuer(token)
            self.validate_jwt_issuer(unverified_issuer)

            auth_server = self._get_auth_server_by_issuer(unverified_issuer)
            jwks_uri = auth_server.metadata.jwks_uri

            if not jwks_uri:
                raise MCPAuthAuthServerException(
                    AuthServerExceptionCode.MISSING_JWKS_URI,
                    cause={
                        "error_description": f"The authorization server ('{unverified_issuer}') does not have a JWKS URI configured."
                    },
                )

            verify_function = create_verify_jwt(jwks_uri, leeway=leeway)
            return verify_function(token)

        return verify_jwt 
