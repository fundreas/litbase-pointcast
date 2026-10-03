"""The OpenAPI document is valid, and the published tree matches it exactly."""

from __future__ import annotations

import copy
import json
import re

import pytest
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate

from kickbase_xp.openapi import build_spec
from kickbase_xp.publish import API_VERSION, publish
from kickbase_xp.rankings import build_rankings
from kickbase_xp.train import run_training

from .conftest import NOW, SEASON, seeded_connection


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """One full publish, rankings included, shared by every test here."""
    tmp = tmp_path_factory.mktemp("openapi")
    conn = seeded_connection(tmp / "test.sqlite")
    run = run_training(conn, max_seasons=None, auto_select=False, now=NOW)
    rankings = build_rankings(conn, SEASON, now=NOW)
    conn.close()

    out = tmp / "site"
    publish(
        run,
        out,
        feature_rows=run.feature_rows,
        rankings=rankings,
        base_url="https://example.github.io/kickbase-xp/",
    )
    return out


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spec(site):
    return _load(site / API_VERSION / "openapi.json")


def _strict(schema):
    """Forbid undocumented fields -- in the tests only, never in the spec."""
    if isinstance(schema, dict):
        schema = {k: _strict(v) for k, v in schema.items()}
        if "properties" in schema and "additionalProperties" not in schema:
            schema["unevaluatedProperties"] = False
    elif isinstance(schema, list):
        schema = [_strict(v) for v in schema]
    return schema


def _template_regex(path: str) -> re.Pattern:
    pattern = re.escape(path)
    pattern = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", pattern)
    return re.compile(f"^{pattern}$")


def _spec_path_for(spec, url: str) -> str | None:
    # A concrete path (`current.json`) wins over a template, as in OpenAPI.
    if url in spec["paths"]:
        return url
    matches = [p for p in spec["paths"] if _template_regex(p).match(url)]
    return matches[0] if matches else None


def _published(site):
    for path in sorted((site / API_VERSION).rglob("*.json")):
        yield "/" + path.relative_to(site).as_posix(), path


def test_spec_is_valid_openapi_31(spec):
    assert spec["openapi"] == "3.1.0"
    validate(spec)


def test_spec_is_published_and_advertised(site, spec):
    index = _load(site / API_VERSION / "index.json")
    assert index["endpoints"]["openapi"] == f"/{API_VERSION}/openapi.json"
    landing = (site / "index.html").read_text(encoding="utf-8")
    assert 'rel="service-desc"' in landing and f"{API_VERSION}/openapi.json" in landing


def test_base_url_becomes_the_first_server(spec):
    assert spec["servers"][0]["url"] == "https://example.github.io/kickbase-xp"


def test_without_base_url_the_server_is_relative():
    spec = build_spec(API_VERSION)
    assert [s["url"] for s in spec["servers"]] == [".."]


def test_every_published_file_is_documented_and_valid(site, spec):
    components = _strict(spec["components"])
    checked = 0
    for url, path in _published(site):
        if url == f"/{API_VERSION}/openapi.json":
            continue
        spec_path = _spec_path_for(spec, url)
        assert spec_path, f"{url} is published but not in the spec"
        ref = spec["paths"][spec_path]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        validator = Draft202012Validator(
            {"components": components, "$ref": ref},
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )
        errors = sorted(validator.iter_errors(_load(path)), key=lambda e: list(e.path))
        assert not errors, f"{url}: " + "; ".join(
            f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors[:5]
        )
        checked += 1
    assert checked > 10


def test_every_documented_path_is_published(site, spec):
    urls = [url for url, _ in _published(site)]
    for spec_path in spec["paths"]:
        pattern = _template_regex(spec_path)
        assert any(pattern.match(u) for u in urls), f"{spec_path} documented but never written"


def test_index_endpoints_match_the_spec(site, spec):
    index = _load(site / API_VERSION / "index.json")
    documented = set(spec["paths"]) | {f"/{API_VERSION}/openapi.json"}
    assert set(index["endpoints"].values()) <= documented
    rankings = _load(site / API_VERSION / "rankings" / "index.json")
    assert set(rankings["endpoints"].values()) <= documented


def test_operation_ids_are_unique(spec):
    ids = [op["operationId"] for item in spec["paths"].values() for op in item.values()]
    assert len(ids) == len(set(ids))
