import time
import urllib.robotparser
from typing import Any
from urllib.parse import urlparse

import requests  # type: ignore

BOT_UA = "CosmicFoundryBot/0.0.0 (https://github.com/cosmic-foundry/cosmic-foundry)"
# Standard browser-like UA for research / one-time data ingestion tasks.
STANDARD_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class HTTPClient:
    """HTTP client with dual-identity support for data ingestion.

    Use BOT_UA + respect_robots=True for recurring / automated tasks.
    Use STANDARD_UA + respect_robots=False for one-time research ingestion.
    """

    def __init__(self, user_agent: str = BOT_UA, respect_robots: bool = True):
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self._robot_parsers: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request_time: dict[str, float] = {}

    def _get_robot_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robot_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                r = requests.get(
                    f"{base}/robots.txt",
                    headers={"User-Agent": STANDARD_UA},
                    timeout=10,
                )
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
            except Exception:
                pass
            self._robot_parsers[base] = rp
        return self._robot_parsers[base]

    def _respect_crawl_delay(self, url: str) -> None:
        if not self.respect_robots:
            return
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._get_robot_parser(url)
        delay_val = rp.crawl_delay(self.user_agent)
        if delay_val:
            elapsed = time.time() - self._last_request_time.get(base, 0.0)
            if elapsed < float(delay_val):
                time.sleep(float(delay_val) - elapsed)

    def _check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return
        rp = self._get_robot_parser(url)
        if not rp.can_fetch(self.user_agent, url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        self._respect_crawl_delay(url)

    def _record_request(self, url: str) -> None:
        parsed = urlparse(url)
        self._last_request_time[f"{parsed.scheme}://{parsed.netloc}"] = time.time()

    def get(
        self, url: str, headers: dict[str, str] | None = None, **kwargs: Any
    ) -> requests.Response:
        self._check_robots(url)
        actual_headers = {"User-Agent": self.user_agent}
        if headers:
            actual_headers.update(headers)
        response = requests.get(url, headers=actual_headers, **kwargs)
        self._record_request(url)
        return response

    def post(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        self._check_robots(url)
        actual_headers = {"User-Agent": self.user_agent}
        if headers:
            actual_headers.update(headers)
        response = requests.post(
            url, data=data, json=json, headers=actual_headers, **kwargs
        )
        self._record_request(url)
        return response
