import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request


MAX_SOURCE_BYTES = 32 * 1024 * 1024
ALLOWED_MEDIA_TYPES = {
    "application/json",
    "application/jsonl",
    "application/x-ndjson",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/tab-separated-values",
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def validate_source_url(url, allowed_hosts):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Dataset endpoints must use HTTPS.")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Dataset endpoint must not contain credentials or a fragment.")
    if parsed.port not in (None, 443):
        raise ValueError("Dataset endpoints must use the standard HTTPS port.")
    hostname = parsed.hostname.rstrip(".").lower()
    hosts = {host.rstrip(".").lower() for host in allowed_hosts}
    if hostname not in hosts:
        raise ValueError(f"Host {hostname!r} is not in the explicit source allowlist.")
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as error:
        raise ValueError(f"Could not resolve source host {hostname!r}.") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Dataset sources must resolve only to public IP addresses.")
    return parsed, hostname


def fetch_public_text(url, allowed_hosts, max_bytes=MAX_SOURCE_BYTES, timeout=20):
    if not 1 <= max_bytes <= MAX_SOURCE_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_SOURCE_BYTES}.")
    parsed, hostname = validate_source_url(url, allowed_hosts)
    opener = urllib.request.build_opener(_NoRedirectHandler())
    request = urllib.request.Request(
        urllib.parse.urlunparse(parsed),
        headers={"User-Agent": "Woa-Awe-oo dataset-ingest/0.1"},
    )
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not fetch allowlisted source {hostname!r}.") from error
    with response:
        content_type = response.headers.get_content_type().lower()
        if content_type not in ALLOWED_MEDIA_TYPES:
            raise ValueError(f"Unsupported dataset content type: {content_type}.")
        length = response.headers.get("Content-Length")
        if length:
            try:
                declared_size = int(length)
            except ValueError as error:
                raise ValueError("Source returned an invalid Content-Length.") from error
            if declared_size > max_bytes:
                raise ValueError("Dataset endpoint response exceeds the configured size limit.")
        chunks = []
        size = 0
        while chunk := response.read(min(65536, max_bytes - size + 1)):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Dataset endpoint response exceeds the configured size limit.")
            chunks.append(chunk)
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Dataset endpoint must return UTF-8 text.") from error
    if not text.strip():
        raise ValueError("Dataset endpoint returned an empty document.")
    return text, {"host": hostname, "bytes": size, "content_type": content_type}
