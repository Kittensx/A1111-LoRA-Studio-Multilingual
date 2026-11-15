"""Access boundary for LoRA routes: API Basic auth or a logged-in Gradio user.

Gradio 3.41's session contract is app.auth/app.tokens and the two access-token
cookies. Never treat the mere presence of a cookie as authentication.
"""
from secrets import compare_digest
from typing import Optional
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials


def create_access_dependency(cmd_opts):
    configured = getattr(cmd_opts, 'api_auth', None)
    credentials = {}
    if configured:
        for entry in configured.split(','):
            username, separator, password = entry.partition(':')
            if not separator or not username:
                raise ValueError('Invalid --api-auth entry; expected username:password')
            credentials[username] = password
    basic = HTTPBasic(auto_error=False)

    def require_access(request: Request, supplied: Optional[HTTPBasicCredentials] = Depends(basic)):
        # Browser sessions must not authorize cross-origin state changes. API
        # clients with no Origin header still use the authentication below.
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            expected = (request.url.scheme, request.url.netloc)
            if origin:
                try:
                    actual = urlsplit(origin)
                except ValueError as error:
                    raise HTTPException(status_code=403, detail='Invalid LoRA request origin') from error
                if (actual.scheme, actual.netloc) != expected:
                    raise HTTPException(status_code=403, detail='Cross-origin LoRA write rejected')
            if request.headers.get('sec-fetch-site') == 'cross-site':
                raise HTTPException(status_code=403, detail='Cross-site LoRA write rejected')

        gradio_auth = getattr(request.app, 'auth', None) is not None
        if gradio_auth:
            token = request.cookies.get('access-token') or request.cookies.get('access-token-unsecure')
            if token and getattr(request.app, 'tokens', {}).get(token) is not None:
                return
        if supplied is not None:
            expected_password = credentials.get(supplied.username)
            if expected_password is not None and compare_digest(
                supplied.password.encode('utf-8'), expected_password.encode('utf-8')
            ):
                return
        protected = configured or gradio_auth or getattr(cmd_opts, 'gradio_auth', None) or getattr(cmd_opts, 'gradio_auth_path', None)
        if protected:
            raise HTTPException(status_code=401, detail='LoRA API authentication required', headers={'WWW-Authenticate': 'Basic'})

    return require_access
