"""The frontend: static serving, and the exact API sequence the page drives.

No browser here — the JS is a thin fetch client, so the meaningful contract is
(a) the files are served and reference each other, and (b) the request
sequence the page makes works against the app.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app(
        Settings(
            llm_provider="fake",
            embedding_provider="fake",
            vector_store="memory",
            database_url=None,
        )
    )
    return TestClient(app)


class TestStaticServing:
    def test_root_redirects_to_the_app(self, client: TestClient) -> None:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/ui/"

    def test_the_page_and_its_assets_are_served(self, client: TestClient) -> None:
        page = client.get("/ui/")
        assert page.status_code == 200
        assert "Prashniq" in page.text
        # Every same-origin script/stylesheet the page references must be served
        # (asset names are content-hashed by the React build, so discover them).
        import re

        assets = re.findall(r'(?:src|href)="(/ui/[^"]+|(?!https?:)[^"]+\.(?:js|css))"', page.text)
        assert assets, "the page references no scripts or stylesheets"
        for asset in assets:
            path = asset if asset.startswith("/ui/") else f"/ui/{asset}"
            assert client.get(path).status_code == 200, f"asset not served: {path}"

    def test_the_api_still_answers_beside_the_ui(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200
        assert client.get("/interview-types").status_code == 200


class TestFrontendApiSequence:
    """The exact calls app.js makes, in order."""

    def test_plain_interview_flow(self, client: TestClient) -> None:
        with client:
            modes = client.get("/interview-types").json()
            assert modes and all("display_name" in mode for mode in modes)

            interview = client.post(
                "/interviews",
                json={
                    "interview_type": modes[0]["key"],
                    "candidate_id": "ui-cand",
                    "context_id": None,
                },
            ).json()

            question = client.post(f"/interviews/{interview['interview_id']}/start").json()
            assert {"index", "text", "topic", "difficulty"} <= question.keys()

            answered = client.post(
                f"/interviews/{interview['interview_id']}/answers",
                json={"answer": "A frontend-submitted answer."},
            ).json()
            assert answered["next_question"]["index"] == 2

            client.post(f"/interviews/{interview['interview_id']}/complete")
            report = client.get(f"/interviews/{interview['interview_id']}/report").json()
            assert {"overall_score", "dimension_scores", "evidence"} <= report.keys()

            profile = client.get("/candidates/ui-cand/profile").json()
            assert profile["candidate_id"] == "ui-cand"

    def test_targeted_interview_flow_with_context(self, client: TestClient) -> None:
        with client:
            context = client.post(
                "/candidate-contexts",
                json={
                    "candidate_id": "ui-cand",
                    "resume_text": "Built a RAG system using FAISS.",
                    "job_description_text": "Need RAG experience.",
                },
            ).json()

            interview = client.post(
                "/interviews",
                json={
                    "interview_type": "jd_targeted",
                    "candidate_id": "ui-cand",
                    "context_id": context["context_id"],
                },
            )
            assert interview.status_code == 201

    def test_error_details_reach_the_page(self, client: TestClient) -> None:
        # app.js surfaces `detail` from error bodies; make sure it exists.
        response = client.post("/interviews", json={"interview_type": "nonsense"})
        assert response.status_code == 422
        assert "detail" in response.json()

        response = client.get("/interviews/missing/report")
        assert response.status_code == 404
        assert "detail" in response.json()
