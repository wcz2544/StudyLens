"""模型接口出口限制：公开页面只能使用部署者审核过的服务商。"""
import ipaddress
import os
import re
from urllib.parse import urlsplit


DEFAULT_MODEL_BASE_URLS = (
    "https://api.deepseek.com",
    "https://api.openai.com/v1",
    "https://api.siliconflow.cn/v1",
)


def _canonical_base_url(value: str) -> str:
    """只接受无凭据、无端口、无转义的公共 HTTPS 域名基址。"""
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError("模型接口地址为空或过长。")
    raw = value.strip().rstrip("/")
    if any(ord(char) < 33 or ord(char) > 126 for char in raw) or any(c in raw for c in "\\%?#"):
        raise ValueError("模型接口地址不能包含空白、转义、查询参数或片段。")
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname or ""
        if (parsed.scheme != "https" or not host or parsed.username is not None
                or parsed.password is not None or parsed.port is not None
                or parsed.netloc.lower() != host):
            raise ValueError
    except ValueError as exc:
        raise ValueError("模型接口必须使用 HTTPS 公共域名，且不能带账号或端口。") from exc
    # 不允许 IP 字面量和内网命名；浏览器输入不能扩大服务端出口权限。
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("模型接口不能使用 IP 地址。")
    if (not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
            or "." not in host or ".." in host
            or host.endswith((".localhost", ".local", ".internal", ".lan", ".home", ".test", ".invalid"))
            or all(part.isdigit() for part in host.split("."))):
        raise ValueError("模型接口必须使用公共服务商域名。")
    if any(part in {".", ".."} for part in parsed.path.split("/")) or "//" in parsed.path:
        raise ValueError("模型接口路径不正确。")
    return f"https://{host}{parsed.path}"


def allowed_model_base_urls() -> tuple[str, ...]:
    """扩展地址只能由部署者在环境变量中指定，不能从页面请求传入。"""
    extra = os.getenv("STUDYLENS_ALLOWED_MODEL_BASE_URLS", "")
    configured = (*DEFAULT_MODEL_BASE_URLS, *(x.strip() for x in extra.split(",") if x.strip()))
    return tuple(dict.fromkeys(_canonical_base_url(value) for value in configured))


def validate_model_base_url(value: str) -> str:
    base = _canonical_base_url(value)
    if base not in allowed_model_base_urls():
        raise ValueError("此模型接口未获部署者允许，请选择受支持的服务商。")
    return base
