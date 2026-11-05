"""Auxiliary client for OAuth-shaped model-provider plugins (``auth_type`` oauth_external / oauth_device_code).

Built-in OAuth routes (nous, openai-codex, xai-oauth) have their own branches in ``auxiliary_client``.
A plugin OAuth provider reaches this one: its credential is the pooled row ``hermes auth add <name>``
wrote, and its transport comes from ``ProviderProfile.create_client``. ``CredentialPool.select()``
rotates an expiring row before leasing it, so compression and titling never send a dead bearer.
"""

from __future__ import annotations

import contextlib
from typing import Any


def _configured_endpoint(provider: str) -> str:
    """The endpoint the main runtime would use: ``model.base_url`` when ``model.provider`` is this
    provider (a relay / proxy override), else the registered profile's own."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.runtime_provider import _config_base_url_for_provider, _get_model_config
    pconfig = PROVIDER_REGISTRY.get(provider)
    return _config_base_url_for_provider(_get_model_config(), provider) or (pconfig.inference_base_url if pconfig else "")


def is_plugin_oauth_pool(provider: str) -> bool:
    """The provider's pooled OAuth rows refresh through its profile's ``refresh_credential`` hook."""
    from hermes_cli.auth_plugin_providers import plugin_refresh_hook
    return plugin_refresh_hook(provider) is not None


def refresh_pooled_row(provider: str, failed_api_key: str) -> bool:
    """Force-refresh the pooled row that sent ``failed_api_key`` — the refresh the main agent's pool
    performs on a 401. False when no row holds that key: a key the pool never issued is not its to fix,
    and spending another row's single-use refresh token for it would lose that sign-in."""
    from agent import auxiliary_client as aux

    pool = aux._load_pool_with_credentials(provider, " (auth refresh)") if failed_api_key else None
    if pool is None or not pool.entry_id_for_api_key(failed_api_key):
        return False
    return pool.try_refresh_matching(api_key_hint=failed_api_key) is not None


@contextlib.contextmanager
def attribute_failure(client: Any):
    """Stamp a failed request's exception with the bearer that sent it, so recovery acts on the row and
    token generation that actually failed rather than one looked up before a refresh or retry."""
    try:
        yield
    except Exception as exc:
        with contextlib.suppress(AttributeError, TypeError):  # a slotted exception type keeps its error
            exc.hermes_aux_api_key = getattr(client, "api_key", None)
        raise


def sent_api_key(exc: Exception, client: Any = None) -> str:
    """The bearer ``exc``'s request sent (stamped by :func:`attribute_failure`), else ``client``'s key."""
    for key in (getattr(exc, "hermes_aux_api_key", None), getattr(client, "api_key", None)):
        if isinstance(key, str) and key:
            return key
    return ""


def shared_host_pool(resolved_provider: Any, refresh_provider: str) -> Any:
    """The plugin OAuth pool an ``"auto"`` route's client tag names when no host table can: its host is
    shared with an API-key sibling (generativelanguage serves Solstice and Gemini), so only the tag knows."""
    route = str(resolved_provider or "auto").strip().lower()
    return refresh_provider if route == "auto" and refresh_provider and is_plugin_oauth_pool(refresh_provider) else None


def pool_superseded_key(provider: str, api_key: Any, base_url: Any = "") -> bool:
    """True when a plugin OAuth pool no longer vouches for the session's ``api_key``: the row that held
    it has since been refreshed to a new token, or its stored expiry has passed. Pinning that key would
    send a bearer the pool already knows is dead; leasing from the pool refreshes it first. A key the pool
    never issued is only "refreshed away" when it was sent to the pool's own origin: a proxy at another
    origin with its own credential keeps it (the pool's bearer must never go to that host)."""
    if not isinstance(api_key, str) or not api_key or not is_plugin_oauth_pool(provider):
        return False
    from agent import auxiliary_client as aux
    from agent.credential_pool_plugin import plugin_row_is_expiring
    from utils import base_url_origin

    pool = aux._load_pool_with_credentials(provider, " (session key check)")
    if pool is None:
        return False
    entry_id = pool.entry_id_for_api_key(api_key)
    entry = next((e for e in pool.entries() if e.id == entry_id), None) if entry_id else None
    if entry is not None:
        return plugin_row_is_expiring(entry)
    from hermes_cli.auth import PROVIDER_REGISTRY

    # The issuing origin is the row's stored endpoint or the provider's registered one, never model.base_url:
    # a proxy configured there with its own key is not where the pooled token was issued.
    own = PROVIDER_REGISTRY[provider].inference_base_url if provider in PROVIDER_REGISTRY else ""
    origins = {base_url_origin(str(e.runtime_base_url or own)) for e in pool.entries()}
    origin = base_url_origin(str(base_url or ""))
    return bool(origin) and origin in origins  # no (valid) endpoint known: keep the supplied key


def _row_expired(provider: str, entry: Any) -> bool:
    from agent.credential_pool_plugin import plugin_row_is_expiring

    return (getattr(entry, "expires_at_ms", None) is not None and is_plugin_oauth_pool(provider)
            and plugin_row_is_expiring(entry))


def expired_cache_hint(provider: str, entry: Any, pool: Any = None, session_key: Any = None) -> str:
    """``":expired"`` for a plugin OAuth row past its stored expiry, so the aux client cache key changes.
    The row keeps its token until refreshed; a client cached while it was valid would keep sending it
    and pay a 401 before the refresh. A session key pins the client to its own row, which may not be the
    one the pool would pick, so that row's expiry is the one that counts; once that row is refreshed the key
    matches no row, and the marker must not fall back to the pre-expiry cache key (its client sends it)."""
    if pool is not None and isinstance(session_key, str) and session_key and is_plugin_oauth_pool(provider):
        row_id = pool.entry_id_for_api_key(session_key)
        if not row_id:
            return ":superseded"
        entry = next((e for e in pool.entries() if e.id == row_id), entry)
    return ":expired" if _row_expired(provider, entry) else ""


def fresh_entry(provider: str, pool: Any, entry: Any) -> Any:
    """``entry``, refreshed first when it is a plugin OAuth row past its stored expiry: sending its token
    is a certain 401. A failed refresh returns the row unchanged for the 401 recovery to handle. An
    availability probe never refreshes: answering "resolvable?" must not spend a refresh token."""
    from agent import auxiliary_client as aux

    if aux._aux_probe_active() or not _row_expired(provider, entry):
        return entry
    return pool.try_refresh_matching(credential_id=entry.id) or entry


def resolve_plugin_oauth_client(req: Any) -> tuple[Any, Any]:
    """``(client, model)`` for ``req.provider``, or ``(None, None)`` when signed out or transport-less.

    An explicit key from the main runtime wins, so aux shares the session's bearer.
    """
    from agent import auxiliary_client as aux

    provider = req.provider
    api_key = aux._normalize_api_key(req.explicit_api_key)
    entry = None
    if not api_key:
        _exists, entry = aux._select_pool_entry(provider)
        api_key = str(getattr(entry, "runtime_api_key", "") or "") if entry is not None else ""
    base_url = (req.explicit_base_url or str(getattr(entry, "runtime_base_url", "") or "")
                or _configured_endpoint(provider)).strip().rstrip("/")
    client = aux._api_key_profile_supplied_client(provider, api_key=api_key, base_url=base_url) if api_key else None
    if client is None:
        aux._log_once_debug(aux._LOGGED_UNSUPPORTED_OAUTH_KEYS, provider,
                            "resolve_provider_client: OAuth provider %s has no signed-in credential or "
                            "plugin transport, try 'auto'", provider)
        return None, None
    model = aux._normalize_resolved_model(req.model or aux._get_aux_model_for_provider(provider), provider)
    return aux._route_client(req, client, model)
