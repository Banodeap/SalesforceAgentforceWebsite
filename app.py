import os
import time
import uuid
import logging
from threading import Lock

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agentforce-chatbot")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024

PORT = int(os.getenv("PORT", "5000"))
SF_MY_DOMAIN = os.getenv("SF_MY_DOMAIN", "").rstrip("/")
SF_CLIENT_ID = os.getenv("SF_CLIENT_ID", "")
SF_CLIENT_SECRET = os.getenv("SF_CLIENT_SECRET", "")
SF_AGENT_ID = os.getenv("SF_AGENT_ID", "")
SF_AGENT_API_HOST = os.getenv("SF_AGENT_API_HOST", "api.salesforce.com").strip()
SF_BYPASS_USER = os.getenv("SF_BYPASS_USER", "true").lower() == "true"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Demo storage only. For production/multiple workers, use Redis or a database and
# bind each session to an authenticated/authorized website user.
sessions = {}
sessions_lock = Lock()


def missing_config():
    missing = []
    if not SF_MY_DOMAIN or "YOUR_DOMAIN" in SF_MY_DOMAIN:
        missing.append("SF_MY_DOMAIN")
    if not SF_CLIENT_ID or "PASTE_" in SF_CLIENT_ID:
        missing.append("SF_CLIENT_ID")
    if not SF_CLIENT_SECRET or "PASTE_" in SF_CLIENT_SECRET:
        missing.append("SF_CLIENT_SECRET")
    if not SF_AGENT_ID or "PASTE_" in SF_AGENT_ID:
        missing.append("SF_AGENT_ID")
    return missing


class SalesforceAPIError(Exception):
    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def response_json(response):
    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text[:1000]}
    if not response.ok:
        message = (
            data.get("error_description")
            or data.get("message")
            or data.get("error")
            or f"Salesforce returned HTTP {response.status_code}"
        )
        raise SalesforceAPIError(str(message), response.status_code, data)
    return data


def get_access_token():
    """Get a Salesforce OAuth token using External Client App client credentials."""
    url = f"{SF_MY_DOMAIN}/services/oauth2/token"
    try:
        response = requests.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": SF_CLIENT_ID,
                "client_secret": SF_CLIENT_SECRET,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise SalesforceAPIError(f"Could not reach Salesforce OAuth endpoint: {exc}") from exc

    data = response_json(response)
    token = data.get("access_token")
    if not token:
        raise SalesforceAPIError("OAuth response did not contain access_token.", details=data)
    return token


def agent_api(path, token, method="GET", json_body=None, headers=None):
    url = f"https://{SF_AGENT_API_HOST}/einstein/ai-agent/v1{path}"
    request_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update(headers)
    try:
        response = requests.request(
            method,
            url,
            headers=request_headers,
            json=json_body,
            timeout=125,
        )
    except requests.RequestException as exc:
        raise SalesforceAPIError(f"Could not reach Salesforce Agent API: {exc}") from exc
    return response_json(response)


def create_salesforce_session():
    token = get_access_token()
    external_session_key = str(uuid.uuid4())
    body = {
        "externalSessionKey": external_session_key,
        "instanceConfig": {"endpoint": f"{SF_MY_DOMAIN}/"},
        "bypassUser": SF_BYPASS_USER,
    }
    data = agent_api(
        f"/agents/{SF_AGENT_ID}/sessions",
        token,
        method="POST",
        json_body=body,
    )
    session_id = data.get("sessionId")
    if not session_id:
        raise SalesforceAPIError(
            "Agent API did not return sessionId. Check the start-session response.",
            details=data,
        )
    return {
        "session_id": session_id,
        "sequence_id": 0,
        "updated_at": time.time(),
        "external_session_key": external_session_key,
    }


def get_web_session(session_key):
    with sessions_lock:
        session = sessions.get(session_key)
        if not session:
            return None
        if time.time() - session["updated_at"] > SESSION_TTL_SECONDS:
            sessions.pop(session_key, None)
            return None
        return session


def extract_answer(data):
    """Extract the Inform message text from a synchronous Agent API response."""
    texts = []
    for message in data.get("messages", []) if isinstance(data, dict) else []:
        if not isinstance(message, dict):
            continue
        if message.get("type") not in ("Inform", "Text", None):
            continue

        # Depending on the response type, text can be at message.message or message.message.text.
        value = message.get("message")
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, dict):
            text_value = value.get("text")
            if isinstance(text_value, str) and text_value.strip():
                texts.append(text_value.strip())

        text_value = message.get("text")
        if isinstance(text_value, str) and text_value.strip():
            texts.append(text_value.strip())

    if texts:
        return "\n\n".join(dict.fromkeys(texts))
    if isinstance(data, dict) and isinstance(data.get("message"), str):
        return data["message"]
    return ""


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    missing = missing_config()
    return jsonify({"ok": not missing, "configured": not missing, "missing": missing}), (200 if not missing else 503)


@app.post("/api/session")
@limiter.limit("10 per minute")
def start_session():
    missing = missing_config()
    if missing:
        return jsonify({"error": "Fill these settings in .env: " + ", ".join(missing)}), 503
    try:
        sf_session = create_salesforce_session()
        web_session_key = str(uuid.uuid4())
        with sessions_lock:
            sessions[web_session_key] = sf_session
        return jsonify({"sessionKey": web_session_key})
    except SalesforceAPIError as exc:
        log.error("Create session failed: %s | details=%s", exc, exc.details)
        status = exc.status_code if exc.status_code and 400 <= exc.status_code < 600 else 502
        detail = str(exc) if os.getenv("FLASK_ENV") != "production" else "Check server logs and Salesforce OAuth/Agent API configuration."
        return jsonify({"error": "Could not start an Agentforce session.", "detail": detail}), status


@app.post("/api/chat")
@limiter.limit("20 per minute")
def chat():
    payload = request.get_json(silent=True) or {}
    session_key = payload.get("sessionKey", "")
    message_text = payload.get("message", "")
    if not isinstance(session_key, str) or not session_key:
        return jsonify({"error": "Missing sessionKey. Start a new chat."}), 400
    if not isinstance(message_text, str) or not message_text.strip():
        return jsonify({"error": "Enter a message."}), 400
    message_text = message_text.strip()
    if len(message_text) > 4000:
        return jsonify({"error": "Message is too long (maximum 4,000 characters)."}), 400

    sf_session = get_web_session(session_key)
    if not sf_session:
        return jsonify({"error": "This chat session expired. Start a new chat."}), 410

    try:
        token = get_access_token()
        next_sequence = sf_session["sequence_id"] + 1
        data = agent_api(
            f"/sessions/{sf_session['session_id']}/messages",
            token,
            method="POST",
            json_body={
                "message": {
                    "type": "Text",
                    "sequenceId": next_sequence,
                    "text": message_text,
                }
            },
        )
        sf_session["sequence_id"] = next_sequence
        sf_session["updated_at"] = time.time()
        answer = extract_answer(data)
        return jsonify({
            "answer": answer or "The agent responded, but this demo couldn't identify the text in the response. Inspect the Agent API response in server logs.",
            "messageTypes": [
                item.get("type") for item in data.get("messages", [])
                if isinstance(item, dict) and item.get("type")
            ] if isinstance(data, dict) else [],
        })
    except SalesforceAPIError as exc:
        log.error("Send message failed: %s | details=%s", exc, exc.details)
        status = exc.status_code if exc.status_code and 400 <= exc.status_code < 600 else 502
        detail = str(exc) if os.getenv("FLASK_ENV") != "production" else "Check server logs, API permissions, agent ID, and My Domain configuration."
        return jsonify({"error": "Agentforce could not answer this message.", "detail": detail}), status


@app.delete("/api/session")
def end_session():
    payload = request.get_json(silent=True) or {}
    session_key = payload.get("sessionKey", "")
    if not isinstance(session_key, str) or not session_key:
        return "", 204
    sf_session = get_web_session(session_key)
    if sf_session:
        try:
            token = get_access_token()
            agent_api(
                f"/sessions/{sf_session['session_id']}",
                token,
                method="DELETE",
                headers={"x-session-end-reason": "UserRequest"},
            )
        except SalesforceAPIError as exc:
            log.warning("End session warning: %s | details=%s", exc, exc.details)
        finally:
            with sessions_lock:
                sessions.pop(session_key, None)
    return "", 204


if __name__ == "__main__":
    print(f"Agentforce chatbot starting at http://127.0.0.1:{PORT}")
    if missing_config():
        print("Missing .env values:", ", ".join(missing_config()))
    app.run(host="127.0.0.1", port=PORT, debug=False)
