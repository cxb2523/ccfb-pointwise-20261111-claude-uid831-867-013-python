import pytest
from unittest.mock import Mock, patch
import jwt

from mcpauth.auth.token_verifier import TokenVerifier, validate_token_claims
from mcpauth.config import (
    AuthServerConfig,
    AuthServerType,
    AuthorizationServerMetadata,
)
from mcpauth.exceptions import (
    BearerAuthExceptionCode,
    MCPAuthBearerAuthException,
    MCPAuthTokenVerificationException,
    MCPAuthTokenVerificationExceptionCode,
    AuthServerExceptionCode,
    MCPAuthAuthServerException,
)
from mcpauth.types import AuthInfo


@pytest.fixture
def mock_auth_server_config() -> AuthServerConfig:
    return AuthServerConfig(
        metadata=AuthorizationServerMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/auth",
            token_endpoint="https://auth.example.com/token",
            jwks_uri="https://auth.example.com/.well-known/jwks.json",
            response_types_supported=["code"],
        ),
        type=AuthServerType.OAUTH,
    )


@pytest.fixture
def mock_auth_server_config_no_jwks() -> AuthServerConfig:
    return AuthServerConfig(
        metadata=AuthorizationServerMetadata(
            issuer="https://auth-no-jwks.example.com",
            authorization_endpoint="https://auth-no-jwks.example.com/auth",
            token_endpoint="https://auth-no-jwks.example.com/token",
            response_types_supported=["code"],
        ),
        type=AuthServerType.OAUTH,
    )


def create_test_jwt(issuer: str, key: str = "secret", algorithm: str = "HS256") -> str:
    payload = {"iss": issuer, "sub": "1234567890", "aud": "my-api"}
    return jwt.encode(payload, key, algorithm=algorithm)


def test_token_verifier_init(mock_auth_server_config: AuthServerConfig):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    assert verifier._auth_servers == [mock_auth_server_config] # type: ignore[reportProtectedUsage]
    assert verifier._issuers == {"https://auth.example.com"} # type: ignore[reportProtectedUsage]


def test_validate_jwt_issuer_valid(mock_auth_server_config: AuthServerConfig):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    verifier.validate_jwt_issuer("https://auth.example.com")  # Should not raise


def test_validate_jwt_issuer_invalid(mock_auth_server_config: AuthServerConfig):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    with pytest.raises(MCPAuthBearerAuthException) as excinfo:
        verifier.validate_jwt_issuer("https://invalid.example.com")
    assert excinfo.value.code == BearerAuthExceptionCode.INVALID_ISSUER


def test_get_unverified_jwt_issuer():
    verifier = TokenVerifier(auth_servers=[])
    token = create_test_jwt(issuer="https://auth.example.com")
    issuer = verifier._get_unverified_jwt_issuer(token) # type: ignore[reportProtectedUsage]
    assert issuer == "https://auth.example.com"


def test_get_unverified_jwt_issuer_malformed():
    verifier = TokenVerifier(auth_servers=[])
    with pytest.raises(MCPAuthTokenVerificationException) as excinfo:
        verifier._get_unverified_jwt_issuer("not-a-jwt") # type: ignore[reportProtectedUsage]
    assert excinfo.value.code == MCPAuthTokenVerificationExceptionCode.INVALID_TOKEN


def test_get_unverified_jwt_issuer_no_iss():
    verifier = TokenVerifier(auth_servers=[])
    payload = {"sub": "1234567890", "aud": "my-api"}
    token = jwt.encode(payload, "secret", algorithm="HS256")
    with pytest.raises(MCPAuthTokenVerificationException) as excinfo:
        verifier._get_unverified_jwt_issuer(token) # type: ignore[reportProtectedUsage]
    assert excinfo.value.code == MCPAuthTokenVerificationExceptionCode.INVALID_TOKEN


def test_get_auth_server_by_issuer(mock_auth_server_config: AuthServerConfig):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    server = verifier._get_auth_server_by_issuer("https://auth.example.com") # type: ignore[reportProtectedUsage]
    assert server == mock_auth_server_config


def test_get_auth_server_by_issuer_invalid(mock_auth_server_config: AuthServerConfig):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    with pytest.raises(MCPAuthBearerAuthException) as excinfo:
        verifier._get_auth_server_by_issuer("https://invalid.example.com") # type: ignore[reportProtectedUsage]
    assert excinfo.value.code == BearerAuthExceptionCode.INVALID_ISSUER


@patch("mcpauth.auth.token_verifier.create_verify_jwt")
def test_create_verify_jwt_function(
    mock_create_verify_jwt: Mock, mock_auth_server_config: AuthServerConfig
):
    mock_verify_function = Mock(return_value={"sub": "user123"})
    mock_create_verify_jwt.return_value = mock_verify_function

    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    verify_jwt_func = verifier.create_verify_jwt_function()

    token = create_test_jwt(issuer="https://auth.example.com")
    auth_info = verify_jwt_func(token)

    mock_create_verify_jwt.assert_called_once_with(
        "https://auth.example.com/.well-known/jwks.json", leeway=60
    )
    mock_verify_function.assert_called_once_with(token)
    assert auth_info == {"sub": "user123"}


@patch("mcpauth.auth.token_verifier.create_verify_jwt")
def test_create_verify_jwt_function_invalid_issuer(
    mock_create_verify_jwt: Mock, mock_auth_server_config: AuthServerConfig
):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config])
    verify_jwt_func = verifier.create_verify_jwt_function()
    token = create_test_jwt(issuer="https://invalid.example.com")

    with pytest.raises(MCPAuthBearerAuthException) as excinfo:
        verify_jwt_func(token)
    assert excinfo.value.code == BearerAuthExceptionCode.INVALID_ISSUER
    mock_create_verify_jwt.assert_not_called()


@patch("mcpauth.auth.token_verifier.create_verify_jwt")
def test_create_verify_jwt_function_no_jwks_uri(
    mock_create_verify_jwt: Mock, mock_auth_server_config_no_jwks: AuthServerConfig
):
    verifier = TokenVerifier(auth_servers=[mock_auth_server_config_no_jwks])
    verify_jwt_func = verifier.create_verify_jwt_function()
    token = create_test_jwt(issuer="https://auth-no-jwks.example.com")

    with pytest.raises(MCPAuthAuthServerException) as excinfo:
        verify_jwt_func(token)
    assert excinfo.value.code == AuthServerExceptionCode.MISSING_JWKS_URI
    mock_create_verify_jwt.assert_not_called() 


def _make_auth_info(
    issuer: str = "https://auth.example.com",
    audience="my-api",
    scopes=None,
) -> AuthInfo:
    return AuthInfo(
        token="token",
        issuer=issuer,
        client_id="client-id",
        subject="subject-id",
        audience=audience,
        scopes=scopes if scopes is not None else ["read", "write"],
        claims={},
    )


class TestValidateTokenClaims:
    def test_should_pass_when_nothing_is_configured(self):
        # Backward compatibility: all checks stay disabled when not configured
        validate_token_claims(_make_auth_info(issuer="https://anything.example.com"))

    def test_should_pass_when_all_claims_match(self):
        validate_token_claims(
            _make_auth_info(),
            issuer="https://auth.example.com",
            audience="my-api",
            required_scopes=["read", "write"],
        )

    def test_jwt_issuer_exact_match_no_trailing_slash_tolerance(self):
        with pytest.raises(MCPAuthBearerAuthException) as excinfo:
            validate_token_claims(
                _make_auth_info(issuer="https://auth.example.com/"),
                issuer="https://auth.example.com",
            )
        assert excinfo.value.code == BearerAuthExceptionCode.INVALID_ISSUER

    def test_jwt_issuer_mismatch(self):
        with pytest.raises(MCPAuthBearerAuthException) as excinfo:
            validate_token_claims(
                _make_auth_info(issuer="https://evil.example.com"),
                issuer="https://auth.example.com",
            )
        assert excinfo.value.code == BearerAuthExceptionCode.INVALID_ISSUER

    def test_jwt_audience_string_match(self):
        validate_token_claims(_make_auth_info(audience="my-api"), audience="my-api")

    def test_jwt_audience_list_intersection(self):
        validate_token_claims(
            _make_auth_info(audience=["other-api", "my-api"]), audience="my-api"
        )

    def test_jwt_audience_mismatch(self):
        for token_audience in ("other-api", ["other-api"], None):
            with pytest.raises(MCPAuthBearerAuthException) as excinfo:
                validate_token_claims(
                    _make_auth_info(audience=token_audience), audience="my-api"
                )
            assert excinfo.value.code == BearerAuthExceptionCode.INVALID_AUDIENCE

    def test_scope_check_requires_all_required_scopes(self):
        validate_token_claims(_make_auth_info(), required_scopes=["read"])
        with pytest.raises(MCPAuthBearerAuthException) as excinfo:
            validate_token_claims(
                _make_auth_info(scopes=["read"]),
                required_scopes=["read", "write"],
            )
        assert excinfo.value.code == BearerAuthExceptionCode.MISSING_REQUIRED_SCOPES
        assert excinfo.value.cause is not None
        assert excinfo.value.cause.missing_scopes == ["write"]

    def test_scope_check_disabled_for_empty_list(self):
        # An empty required-scopes list means "no scopes required" — it must
        # neither deny access nor demand any particular scope.
        validate_token_claims(_make_auth_info(scopes=[]), required_scopes=[])
        validate_token_claims(_make_auth_info(scopes=["read"]), required_scopes=[])
