import os
import re
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
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

# =========================
# OCR (OCR.space)
# =========================
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "none").lower()
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY", "")
OCR_LANG = os.getenv("OCR_LANG", "spa")  # OCR.space usa: spa, eng, etc.

async def ocr_via_ocrspace(file: UploadFile) -> str:
    if not OCRSPACE_API_KEY:
        raise HTTPException(status_code=500, detail="OCRSPACE_API_KEY no configurada")

    content = await file.read()

    url = "https://api.ocr.space/parse/image"
    data = {
        "apikey": OCRSPACE_API_KEY,
        "language": OCR_LANG,
        "isOverlayRequired": "false",
        "OCREngine": "2",
    }

    files = {
        "filename": (
            file.filename or "image.jpg",
            content,
            file.content_type or "application/octet-stream",
        )
    }

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, data=data, files=files)

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OCR.space error {r.status_code}: {r.text}")

    j = r.json()

    try:
        parsed_results = j.get("ParsedResults", [])
        if not parsed_results:
            return ""
        text_out = parsed_results[0].get("ParsedText", "") or ""
        return text_out.strip()
    except Exception:
        return ""


# =========================
# CONFIG
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

SECRET_KEY = os.getenv("SECRET_KEY", "change_me")
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


def ensure_schema():
    """Migración simple: añadir email si no existe"""
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


class HiddenCategory(Base):
    __tablename__ = "hidden_categories"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_hidden_category"),)

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
    date: Optional[str] = None
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


amount_anywhere_re = re.compile(r"(-?\d+(?:[.,]\d{1,2})?)\s*€?")

def _fallback_items_from_text(lines: List[str]) -> List[ParsedExpenseItem]:
    items: List[ParsedExpenseItem] = []
    clean = []
    for ln in lines:
        s = (ln or "").strip()
        if not s or len(s) <= 1:
            continue
        clean.append(s)

    for ln in clean:
        # pillamos el primer importe que aparezca
        m = amount_anywhere_re.search(ln)
        if not m:
            continue

        raw = m.group(1).replace(",", ".")
        try:
            amount = float(raw)
        except:
            continue

        desc = ln.replace(m.group(0), "").strip(" -:\t")
        if not desc:
            desc = "Gasto"

        items.append(
            ParsedExpenseItem(
                date=_today_iso(),
                description=desc[:120],
                amount=abs(amount),
                category=DEFAULT_CATEGORIES[0],
                extra="",
                confidence=0.20,
            )
        )
    return items


def _normalize_date(d: str) -> Optional[str]:
    d = (d or "").strip()

    # yyyy-mm-dd
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", d)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}T00:00:00"

    # dd/mm/yyyy o dd-mm-yyyy
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", d)
    if m:
        dd, mm, yy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        yyyy = ("20" + yy) if len(yy) == 2 else yy
        return f"{yyyy}-{mm}-{dd}T00:00:00"

    # dd/mm (sin año) -> año actual
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", d)
    if m:
        dd, mm = m.group(1).zfill(2), m.group(2).zfill(2)
        yyyy = str(datetime.utcnow().year)
        return f"{yyyy}-{mm}-{dd}T00:00:00"

    return None


def _find_amount(text: str) -> Optional[float]:
    # soporta "12,34", "12.34", "12€", "12,34 €"
    m = re.search(r"(-?\d+(?:[.,]\d{1,2})?)\s*€?", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except:
        return None


def _pick_category(text: str) -> str:
    t = (text or "").lower()
    for c in DEFAULT_CATEGORIES:
        if c.lower() in t:
            return c
    # heurística básica por keywords (opcional, puedes quitarlo si no quieres)
    if any(k in t for k in ["metro", "uber", "bus", "renfe", "taxi"]):
        return "Transporte"
    if any(k in t for k in ["mercadona", "carrefour", "aldi", "lidl", "super", "compra"]):
        return "Compra/Supermercado"
    if any(k in t for k in ["cafe", "caf", "bar", "desay", "tost", "menu", "comida", "cena"]):
        return "Desayuno/Fuera"
    if any(k in t for k in ["spotify", "netflix", "prime", "chatgpt", "suscrip"]):
        return "Suscripciones"
    if any(k in t for k in ["tabaco", "cigar", "vaper"]):
        return "Tabaco"
    if any(k in t for k in ["cerve", "vino", "alcohol"]):
        return "Alcohol/Cervezas"

    return DEFAULT_CATEGORIES[0]


def parse_text_to_items(text: str) -> List[ParsedExpenseItem]:
    """
    Intenta sacar: fecha + descripción + importe + categoría + extra
    - Si el OCR viene por líneas con info dispersa, lo agrupa por proximidad.
    """
    raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not raw_lines:
        return []

    items: List[ParsedExpenseItem] = []

    # Estrategia:
    # 1) detecta líneas que tengan importe => candidato a item
    # 2) para cada candidato, busca fecha cerca (misma línea o líneas anteriores)
    # 3) limpia descripción quitando fecha/importe/categoría
    for i, ln in enumerate(raw_lines):
        amount = _find_amount(ln)
        if amount is None:
            continue

        # Buscar fecha en la misma línea o hasta 3 líneas hacia arriba
        date_iso = _normalize_date(ln)
        if not date_iso:
            for j in range(max(0, i - 3), i):
                date_iso = _normalize_date(raw_lines[j])
                if date_iso:
                    break
        if not date_iso:
            date_iso = _today_iso()

        # Categoría
        cat = _pick_category(ln)

        # Descripción y extra
        desc = ln

        # quita fecha si existe
        desc = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", desc)
        desc = re.sub(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", "", desc)

        # quita importe
        desc = re.sub(r"(-?\d+(?:[.,]\d{1,2})?)\s*€?", "", desc)

        # quita la categoría escrita (si estaba)
        desc = desc.replace(cat, "").strip(" -:\t")

        if not desc:
            # si la línea del importe era muy “seca”, usa algo de líneas previas como descripción
            prev = raw_lines[i - 1] if i - 1 >= 0 else ""
            desc = prev[:120] if prev else "Gasto"

        # extra: si hay “ - ” o “ | ” lo que quede al final
        extra = ""
        if " - " in ln:
            parts = ln.split(" - ", 1)
            if len(parts) == 2:
                extra = parts[1].strip()[:120]

        items.append(
            ParsedExpenseItem(
                date=date_iso,
                description=desc[:120],
                amount=abs(float(amount)),
                category=cat[:64],
                extra=extra,
                confidence=0.35,
            )
        )

    # Si no encontró ninguna línea con importe, intenta un único item del texto entero
    if not items:
        amount = _find_amount(text)
        if amount is None:
            return []
        date_iso = _normalize_date(text) or _today_iso()
        cat = _pick_category(text)
        desc = (text.strip()[:120] if text.strip() else "Gasto")
        items = [
            ParsedExpenseItem(
                date=date_iso,
                description=desc,
                amount=abs(float(amount)),
                category=cat,
                extra="",
                confidence=0.20,
            )
        ]

    return items


def _detect_image_kind(data: bytes) -> str:
    """
    Detecta jpg/png/webp por magic bytes.
    Devuelve: 'jpeg' | 'png' | 'webp' | ''
    """
    if not data or len(data) < 12:
        return ""

    # JPEG: FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"

    # WEBP: "RIFF....WEBP"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"

    return ""


def _ext_from_filename(name: str) -> str:
    name = (name or "").lower().strip()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "jpeg"
    if name.endswith(".png"):
        return "png"
    if name.endswith(".webp"):
        return "webp"
    return ""


# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kikeastro28.github.io",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
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

    user = User(nickname=data.nickname, hashed_password=hash_password(data.password))
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
# CATEGORIES
# =========================
@app.get("/categories", response_model=List[str])
def list_categories(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Category).filter(Category.user_id == user.id).order_by(Category.name.asc()).all()
    custom = [r.name for r in rows]

    hidden_rows = db.query(HiddenCategory).filter(HiddenCategory.user_id == user.id).all()
    hidden = {r.name.strip().lower() for r in hidden_rows}

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
def add_category(payload: CategoryIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
        return {"ok": True}

    db.add(Category(user_id=user.id, name=name))
    db.commit()
    return {"ok": True}


@app.delete("/categories/{name}")
def delete_category(name: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    name = unquote(name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Empty category")

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
def hide_category(payload: CategoryIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
def unhide_category(payload: CategoryIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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


# =========================
# EXPENSES
# =========================
@app.get("/expenses", response_model=List[ExpenseOut])
def list_expenses(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = (
        db.query(Expense)
        .filter(Expense.user_id == user.id)
        .order_by(Expense.date.desc())
        .all()
    )
    return [expense_to_out(e) for e in items]


@app.post("/expenses", response_model=ExpenseOut)
def add_expense(payload: ExpenseIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
def add_expenses_bulk(payload: List[ExpenseIn], user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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


@app.put("/expenses/{expense_id}", response_model=ExpenseOut)
def update_expense(expense_id: int, payload: ExpenseUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
def delete_expense(expense_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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


# =========================
# ME
# =========================
@app.get("/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)):
    return MeResponse(nickname=user.nickname, email=user.email)


@app.post("/me/email")
def update_email(payload: UpdateEmailRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    email = payload.email.strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    user.email = email
    db.add(user)
    db.commit()
    return {"ok": True}


@app.post("/me/change-password")
def change_password(payload: ChangePasswordRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password incorrect")
    user.hashed_password = hash_password(payload.new_password)
    db.add(user)
    db.commit()
    return {"ok": True}


# =========================
# PARSE
# =========================
@app.post("/parse/text", response_model=ParseResponse)
def parse_text(payload: ParseTextRequest, user: User = Depends(get_current_user)):
    items = parse_text_to_items(payload.text)
    return {"items": items}


@app.post("/parse/image")
async def parse_image(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    # Leemos bytes UNA vez
    data = await file.read()

    # Detecta formato por magic bytes (y si falla, por extensión)
    kind = _detect_image_kind(data)
    if not kind:
        kind = _ext_from_filename(file.filename or "")

    if kind not in ("jpeg", "png", "webp"):
        raise HTTPException(status_code=400, detail="Formato no soportado (jpeg/png/webp)")

    if OCR_PROVIDER != "ocrspace":
        raise HTTPException(status_code=503, detail="OCR no configurado (OCR_PROVIDER!=ocrspace)")

    # Importante: como ya leímos file.read(), recreamos un "UploadFile lógico"
    # para ocr_via_ocrspace: le pasamos el contenido directamente con un wrapper simple.
    class _MemUpload:
        def __init__(self, filename, content_type, content_bytes):
            self.filename = filename
            self.content_type = content_type
            self._b = content_bytes

        async def read(self):
            return self._b

    mem = _MemUpload(file.filename or f"image.{kind}", file.content_type or "application/octet-stream", data)

    text_out = await ocr_via_ocrspace(mem)

    if not text_out.strip():
        return {"items": []}

    items = parse_text_to_items(text_out)
    return {"items": items}


@app.get("/")
def root():
    return {"status": "ok", "service": "backend-gastos"}


@app.head("/")
def root_head():
    return Response(status_code=200)
