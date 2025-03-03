import pytest
from fastapi.testclient import TestClient
from app.main import app, get_db
from app.database import Base, engine

# Configurar una base de datos de prueba en memoria
@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

# Cliente de prueba
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Test 1: Verificar que la raíz devuelve HTML
def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

# Test 2: Registro de usuario exitoso
def test_register_user(client):
    response = client.post("/register/", json={
        "username": "testuser",
        "password": "testpass"
    })
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

# Test 3: Login y generación de historia
def test_full_flow(client):
    # Login
    login_response = client.post("/login", data={
        "username": "testuser",
        "password": "testpass"
    })
    token = login_response.json()["access_token"]
    
    # Generar historia (mockeando OpenAI)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("openai.ChatCompletion.create", lambda *args, **kwargs: {
            "choices": [{"message": {"content": "Historia de prueba generada"}}]
        })
        
        story_response = client.post(
            "/story",
            json={"prompt": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert story_response.status_code == 200
        assert "Historia de prueba generada" in story_response.json()["historia"]

# Test 4: Error de usuario duplicado
def test_duplicate_user(client):
    response = client.post("/register/", json={
        "username": "testuser",
        "password": "testpass"
    })
    assert response.status_code == 400
    assert "El usuario ya existe" in response.json()["detail"]