from datetime import datetime, timedelta
from typing import List, Optional
import os
from urllib.parse import unquote

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

from passlib.context import CryptContext
from jose import jwt, JWTError

from fastapi import UploadFile, File
import re
# =========================
# CONFIG
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Render a veces usa postgres:// y SQLAlchemy quiere postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

SECRET_KEY = os.getenv("SECRET_KEY", "s8d9f7sdf8s7df98s7df9sdf7sdf98s7df9s8df")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 días

DEFAULT_CATEGORIES = [
    "Desayuno/Fuera",
    "Compra/Supermercado",
    "Alcohol/Cervezas",
    "Regalos",
    "Transporte",
    "Ropa/Complementos",
    "Suscripciones",
    "Tabaco",
]

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

security = HTTPBearer()

# =========================
# DATABASE
# =========================
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# =========================
# MIGRACIÓN SIMPLE (email)
# =========================
def ensure_schema():
    with engine.begin() as conn:
        dialect = engine.dialect.name

        if dialect == "sqlite":
            table = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            ).fetchone()
            if not table:
                return

            cols = conn.execute(text("PRAGMA table_info(users)")).fetchall()
            col_names = {row[1] for row in cols}
            if "email" not in col_names:
                conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR"))
        else:
            table_exists = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'users'
                    )
                    """
                )
            ).scalar()

            if not table_exists:
                return

            col_exists = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='users' AND column_name='email'
                    )
                    """
                )
            ).scalar()

            if not col_exists:
                conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR"))

# =========================
# DB MODELS
# =========================

# =========================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nickname = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)

    email = Column(String, nullable=True, unique=False)

    expenses = relationship("Expense", back_populates="user", cascade="all, delete-orphan")


class Expense(Base):
    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="expenses")

    date = Column(DateTime, nullable=False, default=datetime.utcnow)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    extra = Column(String, nullable=False, default="")


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_category"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)

    user = relationship("User")


Base.metadata.create_all(bind=engine)
ensure_schema()


# =========================
# Pydantic Schemas
# =========================
class RegisterRequest(BaseModel):
    nickname: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=4, max_length=128)


class LoginRequest(BaseModel):
    nickname: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ExpenseIn(BaseModel):
    date: Optional[str] = None  # ISO string
    description: str
    amount: float
    category: str
    extra: str = ""


class ExpenseOut(BaseModel):
    id: int
    date: str
    description: str
    amount: float
    category: str
    extra: str

class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)

class HiddenCategory(Base):
    __tablename__ = "hidden_categories"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_hidden_category"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)

    user = relationship("User")

Base.metadata.create_all(bind=engine)

class ExpenseUpdate(BaseModel):
    date: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    extra: Optional[str] = None


class MeResponse(BaseModel):
    nickname: str
    email: Optional[str] = None

class UpdateEmailRequest(BaseModel):
    email: str = Field(min_length=3, max_length=120)

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=4, max_length=128)
    new_password: str = Field(min_length=4, max_length=128)


class ParseTextRequest(BaseModel):
    text: str = Field(min_length=1)

class ParsedExpenseItem(BaseModel):
    date: str
    description: str
    amount: float
    category: str
    extra: str = ""
    confidence: float = 0.0

class ParseResponse(BaseModel):
    items: List[ParsedExpenseItem]
# =========================
# HELPERS
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)


def create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.id == int(user_id_str)).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def expense_to_out(e: Expense) -> ExpenseOut:
    return ExpenseOut(
        id=e.id,
        date=e.date.isoformat(),
        description=e.description,
        amount=e.amount,
        category=e.category,
        extra=e.extra or "",
    )


def _today_iso():
    return datetime.utcnow().isoformat()

def _simple_amount_guess(text: str):
    m = re.search(r"(\d+(?:[.,]\d{1,2})?)\s*€?", text)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

# =========================
# APP
# =========================
app = FastAPI()

# CORS: para Flutter Web

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://funny-praline-825b00.netlify.app",   # <-- TU NETLIFY (cámbialo por el tuyo)
        "https://backend-gastos-g560.onrender.com",   # opcional
    ],
    allow_origin_regex=r"^http://localhost:\d+$|^http://127\.0\.0\.1:\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# AUTH
# =========================
@app.post("/auth/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.nickname == data.nickname).first()
    if existing:
        raise HTTPException(status_code=400, detail="Nickname already exists")

    user = User(
        nickname=data.nickname,
        hashed_password=hash_password(data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True}


@app.post("/auth/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.nickname == data.nickname).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user.id)
    return TokenResponse(access_token=token)


# =========================
# EXPENSES (PROTEGIDO)
# =========================

@app.get("/categories", response_model=List[str])
def list_categories(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(Category).filter(Category.user_id == user.id).order_by(Category.name.asc()).all()
    custom = [r.name for r in rows]
    
    hidden_rows = db.query(HiddenCategory).filter(HiddenCategory.user_id == user.id).all()
    hidden = {r.name.strip().lower() for r in hidden_rows}


    # unir defaults + custom sin duplicados, preservando defaults primero
    out = []
    seen = set()
    for c in DEFAULT_CATEGORIES + custom:
        k = c.strip()
        if not k:
            continue
        if k.lower() in hidden:
            continue
        if k.lower() in seen:
            continue
        seen.add(k.lower())
        out.append(k)

    return out


@app.post("/categories")

def add_category(
    payload: CategoryIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Empty category")

    exists = (
        db.query(Category)
        .filter(Category.user_id == user.id)
        .filter(Category.name.ilike(name))
        .first()
    )
    if exists:
        return {"ok": True}  # ya existe, no pasa nada

    db.add(Category(user_id=user.id, name=name))
    db.commit()
    return {"ok": True}



@app.get("/expenses", response_model=List[ExpenseOut])
def list_expenses(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items = (
        db.query(Expense)
        .filter(Expense.user_id == user.id)
        .order_by(Expense.date.desc())
        .all()
    )
    return [expense_to_out(e) for e in items]


@app.post("/expenses", response_model=ExpenseOut)
def add_expense(
    payload: ExpenseIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dt = datetime.utcnow()
    if payload.date:
        dt = datetime.fromisoformat(payload.date)

    e = Expense(
        user_id=user.id,
        date=dt,
        description=payload.description,
        amount=float(payload.amount),
        category=payload.category,
        extra=payload.extra or "",
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return expense_to_out(e)


@app.post("/expenses/bulk", response_model=List[ExpenseOut])
def add_expenses_bulk(
    payload: List[ExpenseIn],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    out = []
    for p in payload:
        dt = datetime.utcnow()
        if p.date:
            dt = datetime.fromisoformat(p.date)

        e = Expense(
            user_id=user.id,
            date=dt,
            description=p.description,
            amount=float(p.amount),
            category=p.category,
            extra=p.extra or "",
        )
        db.add(e)
        out.append(e)

    db.commit()
    for e in out:
        db.refresh(e)
    return [expense_to_out(e) for e in out]


@app.delete("/categories/{name}")
def delete_category(
    name: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = unquote(name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Empty category")

    # No borramos categorías por defecto (seguridad)
    if name.lower() in {c.lower() for c in DEFAULT_CATEGORIES}:
        raise HTTPException(status_code=400, detail="Cannot delete default category")

    row = (
        db.query(Category)
        .filter(Category.user_id == user.id)
        .filter(Category.name.ilike(name))
        .first()
    )
    if not row:
        return {"ok": True}

    db.delete(row)
    db.commit()
    return {"ok": True}

@app.post("/categories/hide")
def hide_category(
    payload: CategoryIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Empty category")

    exists = (
        db.query(HiddenCategory)
        .filter(HiddenCategory.user_id == user.id)
        .filter(HiddenCategory.name.ilike(name))
        .first()
    )
    if exists:
        return {"ok": True}

    db.add(HiddenCategory(user_id=user.id, name=name))
    db.commit()
    return {"ok": True}

@app.post("/categories/unhide")
def unhide_category(
    payload: CategoryIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Empty category")

    row = (
        db.query(HiddenCategory)
        .filter(HiddenCategory.user_id == user.id)
        .filter(HiddenCategory.name.ilike(name))
        .first()
    )
    if not row:
        return {"ok": True}

    db.delete(row)
    db.commit()
    return {"ok": True}


@app.put("/expenses/{expense_id}", response_model=ExpenseOut)
def update_expense(
    expense_id: int,
    payload: ExpenseUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    e = (
        db.query(Expense)
        .filter(Expense.id == expense_id)
        .filter(Expense.user_id == user.id)
        .first()
    )
    if not e:
        raise HTTPException(status_code=404, detail="Expense not found")

    if payload.date is not None:
        e.date = datetime.fromisoformat(payload.date)
    if payload.description is not None:
        e.description = payload.description
    if payload.amount is not None:
        e.amount = float(payload.amount)
    if payload.category is not None:
        e.category = payload.category
    if payload.extra is not None:
        e.extra = payload.extra

    db.commit()
    db.refresh(e)
    return expense_to_out(e)


@app.delete("/expenses/{expense_id}")
def delete_expense(
    expense_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    e = (
        db.query(Expense)
        .filter(Expense.id == expense_id)
        .filter(Expense.user_id == user.id)
        .first()
    )
    if not e:
        return {"ok": True}

    db.delete(e)
    db.commit()
    return {"ok": True}

@app.get("/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)):
    return MeResponse(nickname=user.nickname, email=user.email)


@app.post("/me/email")
def update_email(
    payload: UpdateEmailRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    email = payload.email.strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    user.email = email
    db.add(user)
    db.commit()
    return {"ok": True}


@app.post("/me/change-password")
def change_password(
    payload: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password incorrect")

    user.hashed_password = hash_password(payload.new_password)
    db.add(user)
    db.commit()
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "ok", "service": "backend-gastos"}


@app.post("/parse/text", response_model=ParseResponse)
def parse_text(
    payload: ParseTextRequest,
    user: User = Depends(get_current_user),  # si quieres que sea público, quita esta línea
):
    txt = payload.text.strip()
    amount = _simple_amount_guess(txt)

    if amount is None:
        return {"items": []}

    item = ParsedExpenseItem(
        date=_today_iso(),
        description=txt[:120],
        amount=amount,
        category=DEFAULT_CATEGORIES[0],
        extra="",
        confidence=0.3,
    )
    return {"items": [item]}

@app.post("/parse/image", response_model=ParseResponse)
async def parse_image(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),  # si quieres que sea público, quita esta línea
):
    _ = await file.read()
    return {"items": []}