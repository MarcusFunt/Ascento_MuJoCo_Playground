from fastapi import FastAPI


def test_starlette_testclient_executes_fastapi_route():
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
