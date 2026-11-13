import json
import os
import urllib.error
import urllib.request


def api_url():
    port = os.environ.get("WOA_PORT", "8765")
    return f"http://127.0.0.1:{port}"


def request_json(path, payload=None):
    timeout = 15 if payload is None else 180
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url() + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {}
        raise RuntimeError(body.get("error", f"Request failed ({error.code}).")) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Local service request to {api_url()} timed out after {timeout} seconds."
        ) from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise RuntimeError(
                f"Local service request to {api_url()} timed out after {timeout} seconds."
            ) from error
        raise RuntimeError(
            f"Cannot connect to {api_url()}. Start the app with `python -m woa start`."
        ) from error
