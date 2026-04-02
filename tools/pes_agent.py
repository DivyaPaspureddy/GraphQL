# ~/ADKAgent/PESAgent/pes_tool.py
"""
PES Practitioner Search tool.

Key features (business logic unchanged):
- Secret Manager support for PES_CLIENT_ID / PES_CLIENT_SECRET (env fallback preserved).
- Access token cache with expiry (refresh before/at expiry).
- Safe, single retry on 401/403 for GraphQL (token refresh).
- requests.Session with retry/backoff for transient HTTP errors.
- Optional structured JSON logging (LOG_FORMAT=json) with clear, documented fields.
- Expanded redaction (client_id, client_secret, bearer token) in logs.
- Loads GraphQL query template from GCS and injects practitionerSearchParams dynamically:
  * If a placeholder (e.g., __DYNAMIC_PARAMS__) exists → replaced directly.
  * Else: surgically replaces the existing inline object via brace-matching.

All original inputs/outputs and core logic remain the same.
"""

import os
import sys
import json
import base64
import time
import logging
import hashlib
import threading
from typing import Optional, Any, Dict, List, Set
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv, find_dotenv




# --------------------------- Logging utilities --------------------------------

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Thread-safe request id generator (per-process randomness)
_req_lock = threading.Lock()
_req_seq = 0




def _next_request_id() -> str:
    """Generate a short, process-local request id for cross-log correlation."""
    global _req_seq
    with _req_lock:
        _req_seq = (_req_seq + 1) % 10_000_000
        return f"req-{_req_seq:07d}"


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter to produce structured logs without extra deps."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include all custom attributes set via logger.*(..., extra={...})
        for k, v in record.__dict__.items():
            if k in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname",
                "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "funcName",
            ):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def configure_logging_from_env() -> None:
    """
    UI-safe logging:
    - Silent by default unless LOG_FORMAT is 'plain' or 'json'.
    - Write to stderr (not stdout) so chat UIs don't mirror logs as messages.
    - Disable propagation so logs don't bubble to root handlers.
    """
    fmt = os.getenv("LOG_FORMAT", "").strip().lower()  # "" => silent
    level = os.getenv("LOG_LEVEL", "WARNING").strip().upper()

    logger.handlers = []
    logger.setLevel(getattr(logging, level, logging.WARNING))
    logger.propagate = False

    if fmt in ("plain", "json"):
        handler = logging.StreamHandler(stream=sys.stderr)
        if fmt == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    # Quiet common noisy libraries
    for noisy in ("urllib3", "requests", "google", "grpc", "google.auth"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.WARNING)
        lg.propagate = False


# ---------------------- .env loading (dev convenience) -------------------------

local_env = Path(__file__).with_name(".env")
if local_env.exists():
    load_dotenv(local_env, override=True)
load_dotenv(find_dotenv(usecwd=True), override=True)

# --------------------------- BaseTool (ADK shim) -------------------------------

try:
    from google.adk.tools.base_tool import BaseTool
except Exception as e:
    logger.debug("google.adk not available at import: %s", e)

    class BaseTool:  # minimal shim for local dev/imports
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

        def __call__(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

# ------------------------------ Errors -----------------------------------------

class PESError(Exception):
    """Domain-specific errors for PES tool."""

# ------------------------------- Config (env) ----------------------------------

# --- Add this helper near the config section ---
def _must_env(name: str) -> str:
    val = os.getenv(name)
    if not val or not val.strip():
        raise PESError(f"Missing required environment variable: {name}")
    return val.strip()

# --- Replace current config assignments with strict reads ---

PROJECT_ID              = _must_env("GOOGLE_CLOUD_PROJECT")
GRAPHQL_URL             = _must_env("PES_GRAPHQL_URL")
OAUTH_URL               = _must_env("PES_OAUTH_URL")
OAUTH_SCOPE             = _must_env("PES_OAUTH_SCOPE")
OAUTH_CLIENT_AUTH_METHOD= _must_env("PES_OAUTH_CLIENT_AUTH_METHOD").lower()

# Optional OAuth fields (audience/resource) can remain optional:
OAUTH_AUDIENCE          = os.getenv("PES_OAUTH_AUDIENCE", "").strip()
OAUTH_RESOURCE          = os.getenv("PES_OAUTH_RESOURCE", "").strip()

# Upstream signaling and app name: required if your platform depends on them.
PES_UPSTREAM_ENV        = _must_env("PES_UPSTREAM_ENV")
PES_DEFAULT_APP_NAME    = _must_env("PES_DEFAULT_APP_NAME")

# Query template must be in GCS; enforce strictly
PES_GRAPHQL_QUERY_GCS   = _must_env("PES_GRAPHQL_QUERY_GCS")

# Placeholder is optional (template might not use it); keep a default if desired
PES_GRAPHQL_PARAMS_PLACEHOLDER = os.getenv("PES_GRAPHQL_PARAMS_PLACEHOLDER", "__DYNAMIC_PARAMS__")

# Name/specialty parameterization (make these required if your upstream semantics demand it)
PES_NAME_SOURCE_SYSTEM      = _must_env("PES_NAME_SOURCE_SYSTEM")
PES_NAME_CODESET_TYPE       = _must_env("PES_NAME_CODESET_TYPE")
PES_NAME_WILDCARD_PREFIX    = os.getenv("PES_NAME_WILDCARD_PREFIX", "")
PES_NAME_WILDCARD_SUFFIX    = os.getenv("PES_NAME_WILDCARD_SUFFIX", "")     # set to "" if you want literal searches

PES_SPECIALTY_SOURCE_SYSTEM = os.getenv("PES_SPECIALTY_SOURCE_SYSTEM", "ndb")  # can be optional
PES_SPECIALTY_CODESET_TYPE  = os.getenv("PES_SPECIALTY_CODESET_TYPE", "specialty")
PES_SPECIALTY_WILDCARD_PREFIX = os.getenv("PES_SPECIALTY_WILDCARD_PREFIX", "")
PES_SPECIALTY_WILDCARD_SUFFIX = os.getenv("PES_SPECIALTY_WILDCARD_SUFFIX", "")

# Client ID/Secret are intentionally *not* read here; they’re fetched via Secret Manager or env at runtime in _get_client_id/_get_client_secret

# Secret Manager secret resource names (optional)
PES_CLIENT_ID_SECRET = os.getenv("PES_CLIENT_ID_SECRET", "").strip()
PES_CLIENT_SECRET_SECRET = os.getenv("PES_CLIENT_SECRET_SECRET", "").strip()

# --------------------------- Secret Manager ------------------------------------

_sm_client = None
_secret_cache: Dict[str, str] = {}


def _secret_manager_client():
    """Lazy-init Secret Manager client. Only created if secret names are used."""
    global _sm_client
    if _sm_client is None:
        try:
            from google.cloud import secretmanager  # lazy import
        except Exception as e:
            raise PESError(f"google-cloud-secret-manager not available: {e}") from e
        _sm_client = secretmanager.SecretManagerServiceClient()
    return _sm_client


def _access_secret(secret_resource: str) -> str:
    """Access a secret by resource name. Caches values for the process lifetime."""
    if not secret_resource:
        return ""
    if secret_resource in _secret_cache:
        return _secret_cache[secret_resource]
    client = _secret_manager_client()
    t0 = time.perf_counter()
    response = client.access_secret_version(name=secret_resource)
    val = response.payload.data.decode("utf-8")
    dt = (time.perf_counter() - t0) * 1000
    _secret_cache[secret_resource] = val
    logger.info("Secret accessed", extra={"event": "secret_access", "elapsed_ms": round(dt, 1)})
    return val


def _get_client_id() -> str:
    if PES_CLIENT_ID_SECRET:
        return _access_secret(PES_CLIENT_ID_SECRET)
    return os.getenv("PES_CLIENT_ID", "")


def _get_client_secret() -> str:
    if PES_CLIENT_SECRET_SECRET:
        return _access_secret(PES_CLIENT_SECRET_SECRET)
    return os.getenv("PES_CLIENT_SECRET", "")

# ------------------------------ Redaction --------------------------------------

def _redact_secret(text: str) -> str:
    """Redact sensitive tokens from any string prior to logging."""
    if not isinstance(text, str):
        return text
    client_secret = _get_client_secret()
    client_id = _get_client_id()
    # Mask Bearer tokens and Basic headers
    text = text.replace("Bearer ", "Bearer ***")
    text = text.replace("Basic ", "Basic ***")
    # Mask known credentials (best-effort)
    if client_secret:
        text = text.replace(client_secret, "***")
    if client_id:
        text = text.replace(client_id, "***")
    return text


def set_log_level(level: int) -> None:
    logger.setLevel(level)

# ---------------------------- GCS helpers --------------------------------------

def _split_gs_uri(gs_uri: str) -> Optional[Dict[str, str]]:
    if not gs_uri or not gs_uri.startswith("gs://"):
        return None
    parts = gs_uri[len("gs://") :].split("/", 1)
    if len(parts) == 1:
        bucket, blob_path = parts[0], ""
    else:
        bucket, blob_path = parts[0], parts[1]
    if not bucket or not blob_path:
        return None
    return {"bucket": bucket, "path": blob_path}


def _load_text_from_gcs(gs_uri: Optional[str]) -> Optional[str]:
    if not gs_uri:
        return None
    info = _split_gs_uri(gs_uri)
    if not info:
        logger.warning("Invalid GCS URI for text", extra={"event": "gcs_invalid_uri", "uri": gs_uri})
        return None
    bucket_name, blob_path = info["bucket"], info["path"]

    try:
        from google.cloud import storage  # lazy import
    except Exception as e:
        logger.error("google-cloud-storage not importable", extra={"event": "gcs_import_error", "error": str(e)})
        return None

    try:
        t0 = time.perf_counter()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            logger.error("GCS object not found", extra={"event": "gcs_not_found", "bucket": bucket_name, "path": blob_path})
            return None
        content = blob.download_as_text(encoding="utf-8")
        dt = (time.perf_counter() - t0) * 1000
        logger.info(
            "Loaded text from GCS",
            extra={"event": "gcs_download", "bucket": bucket_name, "path": blob_path, "elapsed_ms": round(dt, 1)},
        )
        return content
    except Exception as e:
        logger.error(
            "Failed to load text from GCS",
            extra={"event": "gcs_download_error", "bucket": bucket_name, "path": blob_path, "error": str(e)},
        )
        return None


def _coerce_to_query(text_or_json: str) -> str:
    try:
        obj = json.loads(text_or_json)
        if isinstance(obj, dict) and isinstance(obj.get("query"), str):
            logger.info("Detected JSON with 'query' field", extra={"event": "query_json_detected"})
            return obj["query"]
    except json.JSONDecodeError:
        pass
    return text_or_json


def _sha256_short(text: str, length: int = 12) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:length]

# ----------------------------- HTTP sessions -----------------------------------

def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_session = _http_session()

# ------------------------------ OAuth helpers ----------------------------------

def _basic_auth_header(client_id: str, client_secret: str) -> str:
    token = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(token).decode("ascii")


def _token_form_body() -> Dict[str, str]:
    body = {"grant_type": "client_credentials"}
    if OAUTH_SCOPE:
        body["scope"] = OAUTH_SCOPE
    if OAUTH_AUDIENCE:
        body["audience"] = OAUTH_AUDIENCE
    if OAUTH_RESOURCE:
        body["resource"] = OAUTH_RESOURCE
    return body


def _raise_oauth_error(resp: Optional[requests.Response]) -> None:
    try:
        status, text = (resp.status_code if resp else None), (resp.text or "")
    except Exception:
        status, text = None, ""
    snippet = _redact_secret((text or "")[:400])
    raise PESError(f"OAuth token request failed {status}: {snippet}")

def _ensure_ok(resp: requests.Response, request_id: str) -> None:
    if resp.status_code != 200:
        text = _redact_secret(resp.text or "")
        raise PESError(f"PES API failed {resp.status_code}: {text[:400]}")

# ------------------------- Access token (expiry cache) --------------------------

_TOKEN: Dict[str, Any] = {"val": None, "exp": 0.0}
_TOKEN_LOCK = threading.Lock()

def _token_valid(now: float) -> bool:
    return bool(_TOKEN["val"]) and now < float(_TOKEN["exp"]) - 60.0  # 60s skew


def get_access_token() -> str:
    now = time.time()
    with _TOKEN_LOCK:
        if _token_valid(now):
            return _TOKEN["val"]

        client_id = _get_client_id()
        client_secret = _get_client_secret()
        if not client_id or not client_secret:
            raise PESError("Missing PES_CLIENT_ID and/or PES_CLIENT_SECRET (env or Secret Manager).")

        headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
        last: Optional[requests.Response] = None
        req_id = _next_request_id()

        def try_post() -> Optional[str]:
            data = _token_form_body()
            data["client_id"] = client_id
            data["client_secret"] = client_secret
            t0 = time.perf_counter()
            resp = _session.post(OAUTH_URL, data=data, headers=headers, timeout=30)
            nonlocal last
            last = resp
            dt = (time.perf_counter() - t0) * 1000
            logger.info("OAuth (POST creds)",
                        extra={"event": "oauth_request", "status": resp.status_code, "elapsed_ms": round(dt, 1), "request_id": req_id})
            if resp.status_code == 200:
                data_j = resp.json()
                tok = data_j.get("access_token")
                exp_in = int(data_j.get("expires_in", 1800))
                if not tok:
                    raise PESError("OAuth token response missing 'access_token'.")
                _TOKEN["val"] = tok
                _TOKEN["exp"] = now + exp_in
                return tok
            return None

        def try_basic() -> Optional[str]:
            hdrs = dict(headers)
            hdrs["Authorization"] = _basic_auth_header(client_id, client_secret)
            data = _token_form_body()
            t0 = time.perf_counter()
            resp = _session.post(OAUTH_URL, data=data, headers=hdrs, timeout=30)
            nonlocal last
            last = resp
            dt = (time.perf_counter() - t0) * 1000
            logger.info("OAuth (Basic auth)",
                        extra={"event": "oauth_request", "status": resp.status_code, "elapsed_ms": round(dt, 1), "request_id": req_id})
            if resp.status_code == 200:
                data_j = resp.json()
                tok = data_j.get("access_token")
                exp_in = int(data_j.get("expires_in", 1800))
                if not tok:
                    raise PESError("OAuth token response missing 'access_token'.")
                _TOKEN["val"] = tok
                _TOKEN["exp"] = now + exp_in
                return tok
            return None

        if OAUTH_CLIENT_AUTH_METHOD == "post":
            tok = try_post()
            if tok:
                return tok
            _raise_oauth_error(last)
        if OAUTH_CLIENT_AUTH_METHOD == "basic":
            tok = try_basic()
            if tok:
                return tok
            _raise_oauth_error(last)

        for fn in (try_post, try_basic):
            tok = fn()
            if tok:
                return tok
        _raise_oauth_error(last)

# -------------------- Strict GCS query selection (lazy) -------------------------

def _choose_query_strict_gcs_only() -> str:
    """
    Load the GraphQL query strictly from GCS. Accepts raw GraphQL text or JSON {"query":"..."}.
    """
    if not PES_GRAPHQL_QUERY_GCS or not PES_GRAPHQL_QUERY_GCS.startswith("gs://"):
        raise PESError(
            "PES_GRAPHQL_QUERY_GCS must be set to a valid gs:// URI (e.g., gs://ep-agent-plus-bucket/query.json)."
        )
    txt = _load_text_from_gcs(PES_GRAPHQL_QUERY_GCS)
    if not txt:
        raise PESError(
            f"Failed to load GraphQL query from {PES_GRAPHQL_QUERY_GCS}. "
            "Check object existence, bucket IAM, and that the runtime service account has Storage Object Viewer."
        )
    q = _coerce_to_query(txt)
    if not isinstance(q, str) or not q.strip():
        raise PESError(f"GraphQL content at {PES_GRAPHQL_QUERY_GCS} is empty or invalid.")
    logger.info(
        "GraphQL loaded strictly from GCS",
        extra={"event": "gcs_query_loaded", "uri": PES_GRAPHQL_QUERY_GCS, "len": len(q), "sha256_12": _sha256_short(q)},
    )
    return q

@lru_cache(maxsize=1)
def get_fhir_query() -> str:
    return _choose_query_strict_gcs_only()

# ------------------------------ Summarizers ------------------------------------

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _text_from_name(name_obj: Any) -> str:
    if not isinstance(name_obj, dict):
        return ""
    text = name_obj.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    given = _as_list(name_obj.get("given"))
    family = name_obj.get("family")
    parts: List[str] = []
    if given:
        parts.append(" ".join([g for g in given if isinstance(g, str) and g.strip()]))
    if isinstance(family, str) and family.strip():
        parts.append(family.strip())
    return " ".join([p for p in parts if p]).strip()

def _extract_display_name(resource: Dict[str, Any]) -> str:
    if not isinstance(resource, dict):
        return ""
    for key in ("namePrac", "name", "practitionerName", "officialName"):
        raw = resource.get(key)
        if raw:
            name_obj = _as_list(raw)[0]
            if isinstance(name_obj, dict):
                disp = _text_from_name(name_obj)
                if disp:
                    return disp
    for k in ("display", "title", "text"):
        v = resource.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _extract_identifiers(resource: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    if not isinstance(resource, dict):
        return ids
    cand = resource.get("identifier")
    for item in _as_list(cand):
        if isinstance(item, dict):
            v = item.get("value") or item.get("id") or item.get("identifierValue")
            if isinstance(v, str) and v.strip():
                ids.append(v.strip())
    seen: Set[str] = set()
    uniq: List[str] = []
    for v in ids:
        if v not in seen:
            uniq.append(v); seen.add(v)
    return uniq

def _entry_type(entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return "(entry)"
    t1 = entry.get("__typename")
    if isinstance(t1, str) and t1.strip():
        return t1.strip()
    res = entry.get("resource") or {}
    if isinstance(res, dict):
        t2 = res.get("resourceType")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()
        t3 = res.get("__typename")
        if isinstance(t3, str) and t3.strip():
            return t3.strip()
    t4 = entry.get("resourceType")
    if isinstance(t4, str) and t4.strip():
        return t4.strip()
    return "(entry)"

def _entry_full_url_or_id(entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return ""
    full_url = entry.get("fullUrl")
    if isinstance(full_url, str) and full_url.strip():
        return full_url.strip()
    res = entry.get("resource") or {}
    if isinstance(res, dict):
        rid = res.get("id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return ""

_USE_DYNAMIC = os.getenv("PES_DYNAMIC_SUMMARY", "0").strip() == "1"

def _summarize_bundle_default(bundle: Dict[str, Any]) -> str:
    total = bundle.get("total")
    links = bundle.get("link") or []
    next_url = ""
    for l in links:
        if l.get("relation") == "next":
            next_url = l.get("url", "")
            break

    entries = bundle.get("entry") or []
    if not entries:
        return f"PES Results: total={total or 0}. No entries. Next={next_url or 'n/a'}"

    sample_lines: List[str] = []
    for e in entries[:50]:
        t = _entry_type(e)
        fullUrl = e.get("fullUrl", "")
        if t == "Practitioner":
            res = e.get("resource") or {}
            display = _extract_display_name(res) or "(no name)"
            id_vals = _extract_identifiers(res)
            ids_txt = ", ".join(id_vals) if id_vals else "n/a"
            sample_lines.append(f"Practitioner | {display} | IDs={ids_txt} | {fullUrl}".strip())
        elif t == "PractitionerRole":
            res = e.get("resource") or {}
            vals = _extract_identifiers(res)
            ids_txt = ", ".join(vals) if vals else "n/a"
            sample_lines.append(f"PractitionerRole | IDs={ids_txt} | {fullUrl}".strip())
        else:
            sample_lines.append(t or "(entry)")

    return (
        f"PES Results: total={total if total is not None else 'unknown'}; "
        f"sample:\n- " + "\n- ".join(sample_lines) + (f"\nNext: {next_url}" if next_url else "")
    )

def _summarize_bundle_dynamic(bundle: Dict[str, Any]) -> str:
    total = bundle.get("total")
    next_url = ""
    for l in _as_list(bundle.get("link")):
        if isinstance(l, dict) and l.get("relation") == "next":
            nu = l.get("url")
            if isinstance(nu, str):
                next_url = nu
                break

    entries = _as_list(bundle.get("entry"))
    if not entries:
        return f"PES Results: total={total or 0}. No entries."

    lines: List[str] = []
    for e in entries[:5]:
        if not isinstance(e, dict):
            lines.append("(entry)")
            continue

        typ = _entry_type(e)
        res = e.get("resource") or {}
        res = res if isinstance(res, dict) else {}

        name = _extract_display_name(res)
        id_vals = _extract_identifiers(res)
        ids_txt = ", ".join(id_vals) if id_vals else "n/a"
        ref = _entry_full_url_or_id(e)

        fields: List[str] = [typ]
        if name:
            fields.append(name)
        fields.append(f"IDs={ids_txt}")
        if ref:
            fields.append(ref)

        lines.append(" | ".join(fields))

    result = f"PES Results: total={total if total is not None else 'unknown'}; sample:\n- " + "\n- ".join(lines)
    if next_url:
        result += f"\nNext: {next_url}"
    return result

def _summarize_bundle(bundle: Dict[str, Any]) -> str:
    if _USE_DYNAMIC:
        return _summarize_bundle_dynamic(bundle)
    return _summarize_bundle_default(bundle)

# ----------------------- Query param injection helpers --------------------------

def _esc(s: Any) -> str:
    """Minimal escape for GraphQL string literals."""
    s = str(s)
    return s.replace("\\", "\\\\").replace('"', '\\"')
def _build_params_object(
    application_name: str,
    page_offset: int,
    page_limit: int,
    name: Optional[str],
    identifier_searches: Optional[List[Dict[str, str]]],
    specialty: Optional[str] = None,    # <-- NEW
) -> str:
    """
    Build practitionerSearchParams object dynamically.
    """

    # ------------------------
    # Name search block
    # ------------------------
    name_block = ""
    if name:
        cs_value = f"{PES_NAME_WILDCARD_PREFIX}{str(name).lower()}{PES_NAME_WILDCARD_SUFFIX}"
        name_block = f"""
        nameSearches: [
          {{
            sourceSystem: "{_esc(PES_NAME_SOURCE_SYSTEM)}",
            codeSetType: "{_esc(PES_NAME_CODESET_TYPE)}",
            codeSetValue: "{_esc(cs_value)}"
          }}
        ]"""

    # ------------------------
    # Specialty search block
    # ------------------------
    specialty_block = ""
    if specialty:
        sp_value = f"{PES_SPECIALTY_WILDCARD_PREFIX}{str(specialty).lower()}{PES_SPECIALTY_WILDCARD_SUFFIX}"
        specialty_block = f"""
        specialtySearches: [
          {{
            sourceSystem: "{_esc(PES_SPECIALTY_SOURCE_SYSTEM)}",
            codeSetType: "{_esc(PES_SPECIALTY_CODESET_TYPE)}",
            codeSetValue: "{_esc(sp_value)}"
          }}
        ]"""

    # ------------------------
    # Identifier searches block
    # ------------------------
    id_block = ""
    if identifier_searches:
        parts: List[str] = []
        for item in identifier_searches:
            if not isinstance(item, dict):
                continue
            ss = _esc(item.get("sourceSystem", ""))
            cst = _esc(item.get("codeSetType", ""))
            csv = _esc(item.get("codeSetValue", ""))
            parts.append(
                f'{{ sourceSystem: "{ss}", codeSetType: "{cst}", codeSetValue: "{csv}" }}'
            )
        if parts:
            id_block = f"identifierSearches: [{', '.join(parts)}]"

    # ------------------------
    # Final fields
    # ------------------------
    fields_only = f"""
        applicationName: "{_esc(application_name)}"
        pageOffset: {page_offset}
        pageLimit: {page_limit}
        {name_block}
        {specialty_block}
        {id_block}
    """

    # ------------------------
    # Return valid GraphQL object
    # ------------------------
    return "{\n" + fields_only + "\n}"


def _inject_params_into_query(template: str, params_object: str, placeholder: str) -> str:
    """
    Inject params into the loaded template:
    - If placeholder (e.g., __DYNAMIC_PARAMS__) exists, replace it.
    - Else, surgically replace the inline object after 'practitionerSearchParams:' via brace matching.
    """
    if placeholder and placeholder in template:
        return template.replace(placeholder, params_object)

    key = "practitionerSearchParams"
    idx = template.find(key)
    if idx == -1:
        raise PESError("Template missing 'practitionerSearchParams' argument.")

    # Find the colon after key
    colon = template.find(":", idx)
    if colon == -1:
        raise PESError("Malformed template around 'practitionerSearchParams:' (no colon).")

    # Find the opening '{' of the inline object
    i = colon + 1
    n = len(template)
    while i < n and template[i].isspace():
        i += 1
    if i >= n or template[i] != "{":
        raise PESError("Malformed template: expected '{' after 'practitionerSearchParams:'")

    # Brace-match to find the end of the object
    start = i
    depth = 0
    j = start
    while j < n:
        ch = template[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
        j += 1
    else:
        raise PESError("Malformed template: unbalanced braces in 'practitionerSearchParams' object.")

    # Replace the original object with our params_object
    return template[:start] + params_object + template[end + 1:]

# --------------------------- Core implementation --------------------------------

def _pes_practitioner_search_impl(
    context: Any,
    application_name: str = PES_DEFAULT_APP_NAME,
    page_offset: int = 0,
    page_limit: int = 50,
    npi: Optional[str] = None,
    identifier_searches: Optional[List[Dict[str, str]]] = None,
    raw: bool = False,
    name: Optional[str] = None, 
    specialty: Optional[str] = None, # optional pass-through for name-based search
) -> str:
    """
    Executes the Practitioner search against PES GraphQL using a query template loaded from GCS.
    The template may:
      - Contain a placeholder (default: __DYNAMIC_PARAMS__) inside the practitionerSearchParams block, OR
      - Have a fully inline practitionerSearchParams object.
    In both cases, we inject a dynamically built params object at runtime.
    """

    # Maintain existing NPI → identifierSearches mapping (non-breaking)
    if npi and not identifier_searches:
        identifier_searches = [
            {"sourceSystem": "ndb", "codeSetType": "nationalproviderid", "codeSetValue": npi}
        ]
    if identifier_searches is None:
        identifier_searches = []

    # Build params object (GraphQL input)
    params_object = _build_params_object(
        application_name=application_name,
        page_offset=page_offset,
        page_limit=page_limit,
        name=name,
        identifier_searches=identifier_searches,
        specialty=specialty, 
    )

    # Load template and inject
    query_template = get_fhir_query()
    query = _inject_params_into_query(query_template, params_object, PES_GRAPHQL_PARAMS_PLACEHOLDER).strip()

    req_id = _next_request_id()

    # Helpful debug (kept minimal; no secrets)
    logger.debug(
        "PES GraphQL inline query snippet",
        extra={"event": "graphql_inline_query", "snippet": query[:900], "request_id": req_id},
    )

    # Safe, minimal request logging (avoid PII leak)
    logger.debug(
        "PES GraphQL request",
        extra={
            "event": "graphql_request",
            "url": GRAPHQL_URL,
            "app": application_name,
            "offset": page_offset,
            "limit": page_limit,
            "npi_present": bool(npi),
            "identifiers_count": len(identifier_searches or []),
            "request_id": req_id,
        },
    )

    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Upstream-Env": PES_UPSTREAM_ENV,
    }

    # PES does not support variables — send only the inline query
    payload = {"query": query}
    t0 = time.perf_counter()
    try:
        resp = _session.post(GRAPHQL_URL, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        logger.error("PES network error", extra={"event": "graphql_network_error", "error": str(e), "request_id": req_id})
        raise PESError(f"PES network error: {e}") from e
    dt = (time.perf_counter() - t0) * 1000

    corr_id = resp.headers.get("x-correlation-id") or resp.headers.get("x-request-id") or ""

    # If token expired/invalid, refresh once and retry the POST
    # ---- Final response handling ----
# ---- Final response handling ----
    _ensure_ok(resp, req_id)

    try:
        body = resp.json()
    except json.JSONDecodeError:
        raise PESError("PES API returned non-JSON response.")

    # ---- GraphQL-level errors ----
    if isinstance(body, dict) and body.get("errors"):
        err0 = body["errors"][0]
        msg = err0.get("message")
        logger.error(
            "GraphQL error",
            extra={
                "event": "graphql_error",
                "error": str(msg),
                "request_id": req_id,
            },
        )
        raise PESError(f"GraphQL error: {msg}")

    # ---- Optional raw diagnostics (never enable in prod) ----
    ALLOW_RAW = os.getenv("PES_TOOL_ALLOW_RAW", "0") == "1"
    if raw and ALLOW_RAW:
        return json.dumps(body, indent=2)[:20000]

    # ---- Extract bundle ----
    bundle = (body.get("data") or {}).get("practitionerSearch")
    if not bundle:
        logger.warning(
            "Empty bundle in response",
            extra={
                "event": "graphql_empty_bundle",
                "request_id": req_id,
            },
        )
        return "PES Results: empty data."

    return _summarize_bundle(bundle)


# ------------------------------ Tool wrapper -----------------------------------
# ------------------------------ Tool wrapper -----------------------------------
class PesPractitionerSearchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="pes_practitioner_search",
            description=(
                "Search PES practitioners (FHIR Bundle). "
                "Args (canonical): application_name, page_offset, page_limit, npi, "
                "identifier_searches, raw, name, specialty. "
                "Also accepts aliases: applicationName (-> application_name), "
                "page_size/limit (-> page_limit)."
            ),
        )
        # Ensure the tool's return is used as the final message
        self.return_direct = True
        # Suppress any model-side narration/summarization of tool calls
        self.skip_summarization = True

        logger.info(
            "Tool init",
            extra={"event": "tool_config", "tool": self.name, "return_direct": self.return_direct}
        )

    def __call__(
        self,
        context: Any,
        application_name: str = PES_DEFAULT_APP_NAME,
        page_offset: int = 0,
        page_limit: int = 50,
        npi: Optional[str] = None,
        identifier_searches: Optional[List[Dict[str, str]]] = None,
        raw: bool = False,
        name: Optional[str] = None,
        specialty: Optional[str] = None,
        # --- Aliases tolerated from Playground / callers ---
        applicationName: Optional[str] = None,   # alias for application_name
        page_size: Optional[int] = None,         # alias for page_limit
        limit: Optional[int] = None,             # alias for page_limit (common)
    ) -> str:
        # Normalize aliases to canonical parameters
        if applicationName and not application_name:
            application_name = applicationName
        if page_size is not None:
            page_limit = page_size
        elif limit is not None:
            page_limit = limit

        # Defensive cleanup: strip '%' (tool manages wildcard '*' via env)
        if isinstance(name, str):
            name = name.replace("%", "").strip()

        return _pes_practitioner_search_impl(
            context=context,
            application_name=application_name,
            page_offset=page_offset,
            page_limit=page_limit,
            npi=npi,
            identifier_searches=identifier_searches,
            raw=raw,
            name=name,
            specialty=specialty,
        )

# Public tool instance
pes_practitioner_search = PesPractitionerSearchTool()
