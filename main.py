import os
import re
import json
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


# =========================
# AI (estructurar OCR -> items)
#   - AI_PROVIDER = "ollama" (local)  o  "openai" (cloud)  o  "none"
# =========================
AI_PROVIDER = os.getenv("AI_PROVIDER", "none").lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama (local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")


def _extract_json_block(s: str) -> str:
    """
    Intenta recuperar el primer bloque JSON (objeto o array) del texto.
    """
    s = (s or "").strip()
    if s.startswith("{") or s.startswith("["):
        return s

    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [x for x in (start_obj, start_arr) if x != -1]
    if not starts:
        return ""
    start = min(starts)
    return s[start:]


async def ai_parse_items_from_text(raw_text: str, allowed_categories: List[str]) -> List["ParsedExpenseItem"]:
    """
    Convierte texto OCR (sucio) en items estructurados usando IA.
    Devuelve lista ParsedExpenseItem con date ISO "YYYY-MM-DDT00:00:00".

    - Si AI_PROVIDER == "ollama": usa /api/chat en OLLAMA_BASE_URL
    - Si AI_PROVIDER == "openai": usa Responses API (Structured Output)
    - Si AI_PROVIDER == "none": devuelve []
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    if AI_PROVIDER == "ollama":
        system = (
            "Eres un asistente que convierte texto OCR de gastos en items estructurados.\n"
            "Devuelve SOLO JSON válido (sin markdown).\n"
            "Formato exacto:\n"
            "{ \"items\": ["
            "{\"date\":\"YYYY-MM-DDT00:00:00\",\"description\":\"...\",\"amount\":12.34,"
            "\"category\":\"...\",\"extra\":\"...\",\"confidence\":0.0}"
            "] }\n"
            "Reglas:\n"
            "- Separa cada gasto en un item distinto.\n"
            "- NO juntes varias descripciones en un solo item.\n"
            "- date: si viene dd/mm/yyyy conviértelo a YYYY-MM-DDT00:00:00.\n"
            "- Si falta fecha, usa la fecha de HOY con T00:00:00.\n"
            "- amount: siempre número con punto decimal.\n"
            "- category: debe ser EXACTAMENTE una de las categorías permitidas.\n"
            "- extra: puede ser \"\".\n"
        )

        user_prompt = (
            "Categorías permitidas:\n"
            + "\n".join(f"- {c}" for c in allowed_categories)
            + "\n\nTexto OCR:\n"
            + raw_text
        )

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        url = f"{OLLAMA_BASE_URL}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(url, json=payload)
        except Exception:
            return []

        if r.status_code != 200:
            return []

        data = r.json()
        content = (((data or {}).get("message") or {}).get("content") or "").strip()
        raw_json = _extract_json_block(content)
        if not raw_json:
            return []

        try:
            obj = json.loads(raw_json)
        except Exception:
            return []

        items = obj.get("items", [])
        out: List[ParsedExpenseItem] = []

        for it in items:
            try:
                date = (it.get("date") or "").strip() or _today_iso_midnight()
                desc = (it.get("description") or "Gasto").strip()
                amount = float(it.get("amount"))
                cat = (it.get("category") or allowed_categories[0]).strip()
                if cat not in allowed_categories:
                    cat = allowed_categories[0]
                extra = (it.get("extra") or "").strip()
                conf = float(it.get("confidence") or 0.7)

                out.append(
                    ParsedExpenseItem(
                        date=date,
                        description=desc[:120],
                        amount=abs(amount),
                        category=cat[:64],
                        extra=extra[:120],
                        confidence=max(0.0, min(conf, 1.0)),
                    )
                )
            except Exception:
                continue

        # filtra basura
        out = [p for p in out if p.amount > 0 and p.date]
        return out

    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

        # JSON Schema estricto (Structured Outputs)
        schema = {
            "name": "parsed_expenses",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "date": {"type": "string"},        # "YYYY-MM-DDT00:00:00"
                                "description": {"type": "string"},
                                "amount": {"type": "number"},
                                "category": {"type": "string"},
                                "extra": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["date", "description", "amount", "category", "extra", "confidence"],
                        },
                    }
                },
                "required": ["items"],
            },
        }

        system = (
            "Eres un extractor de gastos. A partir de texto OCR (posiblemente desordenado), "
            "devuelve SOLO JSON válido que cumpla el esquema. "
            "Reglas:\n"
            "- date SIEMPRE en formato ISO: YYYY-MM-DDT00:00:00\n"
            "- amount en euros (float)\n"
            "- category debe ser UNA de las categorías permitidas\n"
            "- extra puede ser '' si no hay\n"
            "- Separa cada gasto en un item distinto.\n"
            "- No inventes importes.\n"
            "- Si hay filas tipo Excel (fecha | descripción | importe | categoría | extra), respétalas.\n"
        )

        user = (
            "CATEGORÍAS PERMITIDAS:\n"
            + "\n".join(f"- {c}" for c in allowed_categories)
            + "\n\n"
            "TEXTO OCR:\n"
            + raw_text
        )

        payload = {
            "model": OPENAI_MODEL,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)

        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text}")

        data = r.json()

        # Responses API: intenta localizar el texto JSON
        text_json = None
        try:
            text_json = data.get("output_text")
            if not text_json:
                out = data.get("output", [])
                for msg in out:
                    for c in msg.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            text_json = c.get("text")
                            break
                    if text_json:
                        break
        except Exception:
            text_json = None

        if not text_json:
            return []

        try:
            obj = json.loads(text_json)
            items = obj.get("items", [])
            parsed: List[ParsedExpenseItem] = []
            for it in items:
                parsed.append(
                    ParsedExpenseItem(
                        date=(it.get("date") or "").strip() or _today_iso_midnight(),
                        description=(it.get("description") or "").strip()[:120] or "Gasto",
                        amount=float(it.get("amount") or 0),
                        category=(it.get("category") or allowed_categories[0])[:64],
                        extra=(it.get("extra") or "")[:120],
                        confidence=float(it.get("confidence") or 0.6),
                    )
                )
            parsed = [p for p in parsed if p.amount > 0 and p.date]
            return parsed
        except Exception:
            return []

    # AI_PROVIDER == "none" o desconocido
    return []


async def ocr_via_ocrspace(file_like) -> str:
    """
    file_like debe tener:
      - filename
      - content_type
      - async read() -> bytes
    """
    if not OCRSPACE_API_KEY:
        raise HTTPException(status_code=500, detail="OCRSPACE_API_KEY no configurada")

    content = await file_like.read()

    url = "https://api.ocr.space/parse/image"
    data = {
        "apikey": OCRSPACE_API_KEY,
        "language": OCR_LANG,
        "isOverlayRequired": "false",
        "OCREngine": "2",
    }

    files = {
        "filename": (
            getattr(file_like, "filename", None) or "image.jpg",
            content,
            getattr(file_like, "content_type", None) or "application/octet-stream",
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
    date: Optional[str] = None  # "YYYY-MM-DDT00:00:00"
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
    date: str  # "YYYY-MM-DDT00:00:00"
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


def _today_iso_midnight() -> str:
    now = datetime.utcnow()
    return f"{now.year:04d}-{now.month:02d}-{now.day:02d}T00:00:00"


def _normalize_date(d: str) -> Optional[str]:
    d = (d or "").strip()
    if not d:
        return None

    # yyyy-mm-dd
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", d)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}T00:00:00"

    # dd/mm/yyyy o dd-mm-yyyy o dd/mm/yy
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", d)
    if m:
        dd = m.group(1).zfill(2)
        mm = m.group(2).zfill(2)
        yy = m.group(3)
        yyyy = ("20" + yy) if len(yy) == 2 else yy
        return f"{yyyy}-{mm}-{dd}T00:00:00"

    # dd/mm (sin año) -> año actual
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", d)
    if m:
        dd = m.group(1).zfill(2)
        mm = m.group(2).zfill(2)
        yyyy = str(datetime.utcnow().year)
        return f"{yyyy}-{mm}-{dd}T00:00:00"

    return None


def _pick_category(text: str) -> str:
    t = (text or "").lower()

    # si viene ya una categoría literal, la respetamos (solo default aquí)
    for c in DEFAULT_CATEGORIES:
        if c.lower() in t:
            return c

    # heurística por keywords
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


def _is_date_token(tok: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", tok.strip()))


def _is_amount_token(tok: str) -> bool:
    t = tok.strip().replace("€", "")
    # evita tokens que parezcan fecha
    if "/" in t:
        return False
    # el '-' puede aparecer en importes negativos, pero también en fechas; como arriba filtramos '/', aquí vale
    return bool(re.fullmatch(r"-?\d+(?:[.,]\d{1,2})?", t))


def _parse_amount_token(tok: str) -> Optional[float]:
    t = tok.strip().replace("€", "").replace(",", ".")
    try:
        return float(t)
    except Exception:
        return None

DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

def _explode_rows_by_dates(text: str) -> str:
    """
    Si OCR devuelve todo 'aplastado', intentamos reconstruir filas.
    Inserta saltos de línea antes de cada fecha dd/mm/yyyy para que cada fila empiece en una línea.
    """
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()  # aplanado controlado

    # Si hay varias fechas, es muy probable que sea tabla
    dates = DATE_RE.findall(t)
    if len(dates) >= 2:
        # salto de línea antes de cada fecha
        t = DATE_RE.sub(lambda m: "\n" + m.group(0), t).strip()

    return t


def parse_text_to_items(text: str) -> List[ParsedExpenseItem]:
    raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not raw_lines:
        return []

    items: List[ParsedExpenseItem] = []
    current_date_iso: Optional[str] = None

    for ln in raw_lines:
        ln = re.sub(r"\s+", " ", ln).strip()
        tokens = ln.split(" ")
        if not tokens:
            continue

        # 1) fecha al principio (si está)
        if _is_date_token(tokens[0]):
            d_iso = _normalize_date(tokens[0])
            if d_iso:
                current_date_iso = d_iso
            tokens = tokens[1:]
            ln_rest = " ".join(tokens).strip()
        else:
            ln_rest = " ".join(tokens).strip()

        if not ln_rest:
            continue

        # 2) encuentra importe (buscando de derecha a izquierda)
        amount_val = None
        amount_pos = None
        rest_tokens = ln_rest.split(" ")
        for idx in range(len(rest_tokens) - 1, -1, -1):
            if _is_amount_token(rest_tokens[idx]):
                v = _parse_amount_token(rest_tokens[idx])
                if v is not None:
                    amount_val = abs(float(v))
                    amount_pos = idx
                    break

        if amount_val is None or amount_pos is None:
            continue

        before_amount = rest_tokens[:amount_pos]
        after_amount = rest_tokens[amount_pos + 1:]

        # 3) categoría: intenta encontrar una categoría EXACTA en after_amount o en toda la línea
        cat = None
        lower_line = (" ".join(after_amount) + " " + " ".join(before_amount)).lower()
        for c in DEFAULT_CATEGORIES:
            if c.lower() in lower_line:
                cat = c
                break
        if not cat:
            cat = _pick_category(ln_rest)

        # 4) descripción: lo que va antes del importe, limpiando si contiene la categoría
        desc = " ".join(before_amount).strip()
        desc = re.sub(re.escape(cat), "", desc, flags=re.IGNORECASE).strip()
        if not desc:
            desc = "Gasto"

        # 5) extra: lo que queda después del importe, quitando categoría si se repite
        extra = " ".join(after_amount).strip()
        if extra:
            extra = re.sub(re.escape(cat), "", extra, flags=re.IGNORECASE).strip(" -:\t")

        date_iso = current_date_iso or _today_iso_midnight()

        items.append(
            ParsedExpenseItem(
                date=date_iso,
                description=desc[:120],
                amount=amount_val,
                category=cat[:64],
                extra=extra[:120],
                confidence=0.70,
            )
        )

    return items



def _detect_image_kind(data: bytes) -> str:
    if not data or len(data) < 12:
        return ""

    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
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


def get_allowed_categories(user: User, db: Session) -> List[str]:
    """
    default + custom - hidden (sin duplicados, respetando orden)
    """
    custom_rows = (
        db.query(Category)
        .filter(Category.user_id == user.id)
        .order_by(Category.name.asc())
        .all()
    )
    custom = [r.name for r in custom_rows]

    hidden_rows = db.query(HiddenCategory).filter(HiddenCategory.user_id == user.id).all()
    hidden = {r.name.strip().lower() for r in hidden_rows}

    allowed: List[str] = []
    seen = set()
    for c in DEFAULT_CATEGORIES + custom:
        k = (c or "").strip()
        if not k:
            continue
        lk = k.lower()
        if lk in hidden:
            continue
        if lk in seen:
            continue
        seen.add(lk)
        allowed.append(k)

    # fallback mínimo por si algo raro
    return allowed or list(DEFAULT_CATEGORIES)


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
    return get_allowed_categories(user, db)


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


# =========================
# PARSE
# =========================
@app.post("/parse/text", response_model=ParseResponse)
async def parse_text(
    payload: ParseTextRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    allowed = get_allowed_categories(user, db)

    # 1) IA primero (ollama/openai)
    try:
        ai_items = await ai_parse_items_from_text(payload.text, allowed)
        if ai_items:
            return {"items": ai_items}
    except HTTPException:
        # si OpenAI no está configurado, etc.
        pass
    except Exception:
        pass

    # 2) fallback heurístico
    items = parse_text_to_items(payload.text)
    return {"items": items}


@app.post("/parse/image", response_model=ParseResponse)
async def parse_image(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    data = await file.read()

    kind = _detect_image_kind(data) or _ext_from_filename(file.filename or "")
    if kind not in ("jpeg", "png", "webp"):
        raise HTTPException(status_code=400, detail="Formato no soportado (jpeg/png/webp)")

    if OCR_PROVIDER != "ocrspace":
        raise HTTPException(status_code=503, detail="OCR no configurado (OCR_PROVIDER!=ocrspace)")

    class _MemUpload:
        def __init__(self, filename, content_type, content_bytes):
            self.filename = filename
            self.content_type = content_type
            self._b = content_bytes

        async def read(self):
            return self._b

    mem = _MemUpload(
        file.filename or f"image.{kind}",
        file.content_type or "application/octet-stream",
        data,
    )

    text_out = await ocr_via_ocrspace(mem)
    text_out = _explode_rows_by_dates(text_out)

    if not text_out.strip():
        return {"items": []}

    allowed = get_allowed_categories(user, db)

    # 1) IA primero (ollama/openai)
    try:
        ai_items = await ai_parse_items_from_text(text_out, allowed)
        if ai_items:
            return {"items": ai_items}
    except HTTPException:
        pass
    except Exception:
        pass

    # 2) fallback heurístico
    items = parse_text_to_items(text_out)
    return {"items": items}


@app.get("/")
def root():
    return {"status": "ok", "service": "backend-gastos"}


@app.head("/")
def root_head():
    return Response(status_code=200)
