 import os
import uvicorn
import openai
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configuración de la base de datos MySQL en AWS RDS
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_URL = f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelos de base de datos
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    interactions = relationship("Interaction", back_populates="user")

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="interactions")

# Modelos Pydantic
class UserCreate(BaseModel):
    username: str
    email: Optional[EmailStr] = None  # Opcional, si deseas usarlo
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: Optional[EmailStr] = None
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class StoryRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

class StoryResponse(BaseModel):
    historia: str
    timestamp: datetime
    class Config:
        orm_mode = True

# Configuración de seguridad
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(lambda: next(get_db()))):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

# Dependencia para la base de datos
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Inicializar FastAPI y plantillas
app = FastAPI(title="Creador de Historias Interactivo")
templates = Jinja2Templates(directory="templates")  # Asegúrate de crear esta carpeta y colocar index.html

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints

# Página de inicio (puedes usarla para tu frontend)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Registro de usuario
@app.post("/register/", response_model=UserOut)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="El usuario ya existe")
    new_user = User(username=user.username, hashed_password=hash_password(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Login para obtener token JWT
@app.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario o contraseña incorrectos")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint para generar una historia utilizando el LLM de OpenAI
@app.post("/story", response_model=StoryResponse)
def generate_story(request_data: StoryRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Llamada al modelo LLM preentrenado (ejemplo con OpenAI)
    try:
        response = openai.Completion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request_data.prompt}],
            prompt=request_data.prompt,
            max_tokens=request_data.max_tokens,
            temperature=request_data.temperature
        )
        historia = response.choices[0].message['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail= str(e))
    
    # Crear una interacción en la base de datos para almacenar la consulta y respuesta
    interaction = Interaction(
        user_id=current_user.id,
        prompt=request_data.prompt,
        response=historia
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    
    return StoryResponse(historia=historia, timestamp=interaction.created_at)

# Ejecutar la aplicación
if __name__ == "__main__":
    # En desarrollo, ejecuta uvicorn. En producción, se recomienda usar un servidor más robusto.
    uvicorn.run(app, host="0.0.0.0", port=8000)