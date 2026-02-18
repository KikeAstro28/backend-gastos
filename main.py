import os
import re
import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
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
OCR_LANG = os.getenv("OCR_LANG", "spa")  # ocr.space: spa, eng, etc.


async def ocr_via_ocrspace(file_like) -> tuple[str, dict]:
    """
    Devuelve (texto_plano, json_completo) para poder usar TextOverlay (coords).
    """
    if not OCRSPACE_API_KEY:
        raise HTTPException(status_code=500, detail="OCRSPACE_API_KEY no configurada")

    content = await file_like.read()

    url = "https://api.ocr.space/parse/image"
    data = {
        "apikey": OCRSPACE_API_KEY,
        "language": OCR_LANG,
        "isOverlayRequired": "true",   # <-- CLAVE
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
    parsed_results = j.get("ParsedResults", []) or []
    if not parsed_results:
        return "", j

    text_out = (parsed_results[0].get("ParsedText", "") or "").strip()
    return text_out, j


# =========================
# AI (Ollama / OpenAI)
# =========================
AI_PROVIDER = os.getenv("AI_PROVIDER", "none").lower()

# Ollama (local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# OpenAI (prod)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Groq (prod barato)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

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
def _extract_json_obj(text: str) -> Optional[dict]:
    """
    Intenta extraer el primer objeto JSON válido de un texto.
    Soporta respuestas con ```json ... ``` y texto extra.
    """
    if not text:
        return None

    t = text.strip()

    # quita code fences si vienen
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # intento directo
    try:
        return json.loads(t)
    except Exception:
        pass

    # busca primer {...} grande
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None

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


def _is_date_token(tok: str) -> bool:
    tok = tok.strip()
    return bool(re.fullmatch(r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", tok))


def _is_amount_token(tok: str) -> bool:
    t = tok.strip().replace("€", "")
    if "/" in t or "-" in t:
        return False
    return bool(re.fullmatch(r"-?\d+(?:[.,]\d{1,2})?", t))


def _parse_amount_token(tok: str) -> Optional[float]:
    t = tok.strip().replace("€", "").replace(",", ".")
    try:
        return float(t)
    except Exception:
        return None


def _pick_category_from_text(text: str, allowed: List[str]) -> Optional[str]:
    t = (text or "").lower()
    # match exact allowed categories if they appear
    for c in allowed:
        if c.lower() in t:
            return c
    return None


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _limit_description(desc: str, extra: str) -> Tuple[str, str]:
    """
    Regla: descripción corta. Si sale larguísima, recortamos y pasamos resto a extra.
    """
    desc = _clean_spaces(desc)
    extra = _clean_spaces(extra)

    words = desc.split()
    if len(words) <= 8 and len(desc) <= 60:
        return desc, extra

    head = " ".join(words[:8]).strip()
    tail = " ".join(words[8:]).strip()

    if tail:
        extra = _clean_spaces((extra + " " + tail).strip())
    return head[:80], extra[:160]


def explode_candidates_from_ocr(ocr_text: str) -> List[dict]:
    """
    PRE-PARSER robusto:
    - Divide cualquier línea que contenga múltiples fechas en trozos (sin tocar la lista mientras iteras).
    - Detecta fecha en cualquier posición, no solo al principio.
    - Detecta importe como último número válido (evitando confundir día/mes).
    """
    text = (ocr_text or "").strip()
    if not text:
        return []

    # 1) Normaliza y parte en líneas
    raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        return []

    # 2) Si una línea tiene varias fechas, la partimos por cada fecha encontrada
    expanded_lines: List[str] = []
    date_pat = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

    for ln in raw_lines:
        matches = list(date_pat.finditer(ln))
        if len(matches) <= 1:
            expanded_lines.append(ln)
            continue

        # trozos: desde cada fecha hasta la siguiente fecha
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(ln)
            chunk = ln[start:end].strip()
            if chunk:
                expanded_lines.append(chunk)

    candidates: List[dict] = []
    current_date_iso: Optional[str] = None
    pending_parts: List[str] = []

    for ln in expanded_lines:
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln:
            continue

        tokens = ln.split(" ")

        # 3) Busca una fecha en cualquier posición (normalmente al inicio, pero OCR a veces la mueve)
        found_date_iso = None
        date_idx = None
        for idx, tok in enumerate(tokens[:4]):  # normalmente aparece al principio; limitamos para no liarla
            if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", tok):
                found_date_iso = _normalize_date(tok)
                date_idx = idx
                break

        if found_date_iso:
            current_date_iso = found_date_iso
            # quita ese token fecha
            tokens = tokens[:date_idx] + tokens[date_idx + 1 :]

        # 4) Busca el último token que sea importe real
        amount_idx = None
        amount_val = None
        for idx in range(len(tokens) - 1, -1, -1):
            if _is_amount_token(tokens[idx]):
                v = _parse_amount_token(tokens[idx])
                if v is not None and v > 0:
                    amount_idx = idx
                    amount_val = float(v)
                    break

        # si no hay importe, lo tratamos como continuación de texto
        if amount_idx is None or amount_val is None:
            pending_parts.append(" ".join(tokens))
            continue

        before_amount = tokens[:amount_idx]
        after_amount = tokens[amount_idx + 1 :]

        raw = _clean_spaces(" ".join([*pending_parts, " ".join(before_amount), " ".join(after_amount)]))
        pending_parts = []

        date_iso = current_date_iso or _today_iso_midnight()

        candidates.append({"date": date_iso, "amount": amount_val, "raw": raw})

    return candidates

def _is_money_str(s: str) -> bool:
    s = (s or "").strip().replace("€", "").replace(",", ".")
    # evita cosas tipo 02/02/2026
    if "/" in s or "-" in s:
        return False
    return bool(re.fullmatch(r"\d+(?:\.\d{1,2})?", s))

def _to_money(s: str) -> Optional[float]:
    try:
        return float((s or "").strip().replace("€", "").replace(",", "."))
    except Exception:
        return None

def _is_date_str(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s))

def _extract_words_from_ocrspace_overlay(ocr_json: dict) -> list[dict]:
    """
    Devuelve lista de palabras con coords: {text,left,top,width,height}
    """
    try:
        pr = (ocr_json.get("ParsedResults") or [])[0]
        overlay = pr.get("TextOverlay") or {}
        lines = overlay.get("Lines") or []
        out = []
        for ln in lines:
            for w in (ln.get("Words") or []):
                out.append({
                    "text": (w.get("WordText") or "").strip(),
                    "left": int(w.get("Left") or 0),
                    "top": int(w.get("Top") or 0),
                    "width": int(w.get("Width") or 0),
                    "height": int(w.get("Height") or 0),
                })
        return [x for x in out if x["text"]]
    except Exception:
        return []

def _group_words_by_rows(words: list[dict], y_tol: int = 12) -> list[list[dict]]:
    """
    Agrupa palabras por filas según 'top' (coordenada y).
    """
    if not words:
        return []

    ws = sorted(words, key=lambda w: (w["top"], w["left"]))
    rows: list[list[dict]] = []
    for w in ws:
        placed = False
        cy = w["top"]
        for row in rows:
            ry = row[0]["top"]
            if abs(cy - ry) <= y_tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    # ordena cada fila por x
    for row in rows:
        row.sort(key=lambda w: w["left"])
    return rows

def extract_items_from_overlay(
    ocr_json: dict,
    allowed_categories: List[str],
) -> List["ParsedExpenseItem"]:
    """
    Extrae gastos por fila usando coords:
    - amount = número más a la derecha de la fila
    - date = token fecha si aparece en la fila; si no, hereda la última vista
    - description = palabras a la izquierda del amount, recortada a max 4 palabras
    """
    words = _extract_words_from_ocrspace_overlay(ocr_json)
    rows = _group_words_by_rows(words, y_tol=14)

    items: List[ParsedExpenseItem] = []
    current_date_iso: Optional[str] = None

    for row in rows:
        texts = [w["text"] for w in row]

        # detecta fecha en la fila
        date_tok = next((t for t in texts if _is_date_str(t)), None)
        if date_tok:
            d_iso = _normalize_date(date_tok)
            if d_iso:
                current_date_iso = d_iso

        # detecta importes (candidatos)
        money_words = []
        for w in row:
            if _is_money_str(w["text"]):
                val = _to_money(w["text"])
                if val is not None and val > 0:
                    money_words.append((w, val))

        if not money_words:
            continue

        # coge el importe más a la derecha (x mayor)
        money_words.sort(key=lambda t: t[0]["left"])
        amount_word, amount_val = money_words[-1]

        # descripción = palabras claramente a la izquierda del importe
        left_words = [w for w in row if w["left"] + w["width"] <= amount_word["left"] - 8]
        left_text = " ".join(w["text"] for w in left_words).strip()

        # quita fecha de la descripción si se coló
        left_text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "", left_text).strip()

        # si no hay nada a la izquierda, intenta “arriba” (fila anterior cercana)
        desc = left_text
        extra = ""

        if not desc:
            desc = "Gasto"

        # regla: máx 4 palabras para description
        words_desc = desc.split()
        if len(words_desc) > 4:
            desc = " ".join(words_desc[:4])
            extra = " ".join(words_desc[4:])[:120]

        date_iso = current_date_iso or _today_iso_midnight()

        # categoría: si en la fila aparece alguna categoría exacta, úsala; si no, default
        row_join = " ".join(texts).lower()
        cat = next((c for c in allowed_categories if c.lower() in row_join), None)
        if not cat:
            cat = allowed_categories[0] if allowed_categories else DEFAULT_CATEGORIES[0]

        items.append(
            ParsedExpenseItem(
                date=date_iso,
                description=desc[:120],
                amount=float(amount_val),
                category=cat[:64],
                extra=extra[:120],
                confidence=0.80,
            )
        )

    return items


async def ai_refine_candidates(
    candidates: List[dict],
    allowed_categories: List[str],
) -> List[ParsedExpenseItem]:
    """
    IA SOLO refina description/extra y (si falta) category.
    Fecha e importe se respetan del pre-parser.
    """
    print("===== AI DEBUG =====")
    print("AI_PROVIDER =", AI_PROVIDER)
    print("Número de candidatos =", len(candidates))
    print("=====================")

    if not candidates:
        return []

    if AI_PROVIDER not in ("ollama", "openai", "groq"):
        return []


    # Prompt muy explícito para que NO mezcle filas
    system = (
        "Eres un asistente que LIMPIA y NORMALIZA gastos ya detectados.\n"
        "Te doy una lista de CANDIDATOS, cada uno es 1 gasto con date y amount ya fijados.\n"
        "Tu trabajo:\n"
        "1) Para cada candidato, devuelve description corta (max 8 palabras, sin fechas ni importes).\n"
        "2) Devuelve extra breve (opcional) con el resto útil.\n"
        "3) category debe ser UNA de las permitidas. Si el candidato ya contiene una categoría, úsala.\n"
        "4) NO juntes varios candidatos en uno. Mantén el mismo número de items.\n"
        "5) NO inventes importes ni fechas.\n"
    )

    user = {
        "allowed_categories": allowed_categories,
        "candidates": candidates,
        "output_format": {
            "items": [
                {
                    "date": "YYYY-MM-DDT00:00:00",
                    "amount": 0.0,
                    "category": "one-of-allowed",
                    "description": "max 8 words",
                    "extra": "optional",
                    "confidence": 0.0,
                }
            ]
        },
    }

    # --- OLLAMA ---
    if AI_PROVIDER == "ollama":
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, json=payload)

        if r.status_code != 200:
            # si Ollama no está accesible (p.ej. en Render), devolvemos vacío para fallback
            return []

        data = r.json()
        content = (data.get("message") or {}).get("content") or ""
        try:
            obj = _extract_json_obj(content)
            if not obj:
                return []

        except Exception:
            return []

        items = obj.get("items", [])
        out: List[ParsedExpenseItem] = []
        for i, it in enumerate(items):
            # Respetar date/amount del candidato sí o sí
            base = candidates[i] if i < len(candidates) else None
            if not base:
                continue
            date_iso = base["date"]
            amount = float(base["amount"])

            cat = (it.get("category") or "").strip()
            if cat not in allowed_categories:
                # intenta detectar desde raw
                cat2 = _pick_category_from_text(base.get("raw", ""), allowed_categories)
                cat = cat2 or (allowed_categories[0] if allowed_categories else "Desayuno/Fuera")

            desc = (it.get("description") or "").strip() or "Gasto"
            extra = (it.get("extra") or "").strip()

            desc, extra = _limit_description(desc, extra)

            out.append(
                ParsedExpenseItem(
                    date=date_iso,
                    description=desc[:120],
                    amount=amount,
                    category=cat[:64],
                    extra=extra[:120],
                    confidence=float(it.get("confidence") or 0.75),
                )
            )
        # si devolvió menos items, fallback
        if len(out) < max(1, int(0.7 * len(candidates))):
            return []
        return out

    # --- OPENAI ---
    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

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
                                "date": {"type": "string"},
                                "amount": {"type": "number"},
                                "category": {"type": "string"},
                                "description": {"type": "string"},
                                "extra": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["date", "amount", "category", "description", "extra", "confidence"],
                        },
                    }
                },
                "required": ["items"],
            },
        }

        payload = {
            "model": OPENAI_MODEL,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
        }

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)

        if r.status_code != 200:
            return []

        data = r.json()

        # sacar texto json del response
        text_json = data.get("output_text")
        if not text_json:
            outarr = data.get("output", []) or []
            for msg in outarr:
                for c in msg.get("content", []) or []:
                    if c.get("type") in ("output_text", "text"):
                        text_json = c.get("text")
                        break
                if text_json:
                    break

        if not text_json:
            return []

        try:
            obj = json.loads(text_json)
        except Exception:
            return []

        items = obj.get("items", [])
        out: List[ParsedExpenseItem] = []

        for i, it in enumerate(items):
            base = candidates[i] if i < len(candidates) else None
            if not base:
                continue
            date_iso = base["date"]
            amount = float(base["amount"])

            cat = (it.get("category") or "").strip()
            if cat not in allowed_categories:
                cat2 = _pick_category_from_text(base.get("raw", ""), allowed_categories)
                cat = cat2 or (allowed_categories[0] if allowed_categories else "Desayuno/Fuera")

            desc = (it.get("description") or "").strip() or "Gasto"
            extra = (it.get("extra") or "").strip()

            desc, extra = _limit_description(desc, extra)

            out.append(
                ParsedExpenseItem(
                    date=date_iso,
                    description=desc[:120],
                    amount=amount,
                    category=cat[:64],
                    extra=extra[:120],
                    confidence=float(it.get("confidence") or 0.8),
                )
            )

        if len(out) < max(1, int(0.7 * len(candidates))):
            return []
        return out

        # --- GROQ (OpenAI-compatible) ---
    if AI_PROVIDER == "groq":
        print("🔥 Entrando en bloque GROQ")
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada")

        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": GROQ_MODEL,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code != 200:
            return []

        data = r.json()
        content = data["choices"][0]["message"]["content"]

        try:
            obj = _extract_json_obj(content)
            if not obj:
                return []

        except Exception:
            return []

        items = obj.get("items", [])
        out: List[ParsedExpenseItem] = []

        for i, it in enumerate(items):
            base = candidates[i] if i < len(candidates) else None
            if not base:
                continue

            date_iso = base["date"]
            amount = float(base["amount"])

            cat = (it.get("category") or "").strip()
            if cat not in allowed_categories:
                cat2 = _pick_category_from_text(base.get("raw", ""), allowed_categories)
                cat = cat2 or (allowed_categories[0] if allowed_categories else "Desayuno/Fuera")

            desc = (it.get("description") or "").strip()
            if not desc:
                # si IA no dio descripción, la sacamos del raw del candidato
                base_raw = (base.get("raw") or "").strip()
                # quita la categoría si aparece
                if cat:
                    base_raw = re.sub(re.escape(cat), "", base_raw, flags=re.IGNORECASE).strip(" -:\t")
                # deja primeras 6-8 palabras
                words = base_raw.split()
                desc = " ".join(words[:8]).strip() if words else "Gasto"

            extra = (it.get("extra") or "").strip()

            desc, extra = _limit_description(desc, extra)

            out.append(
                ParsedExpenseItem(
                    date=date_iso,
                    description=desc[:120],
                    amount=amount,
                    category=cat[:64],
                    extra=extra[:120],
                    confidence=float(it.get("confidence") or 0.8),
                )
            )

        # si devuelve muy pocos items, consideramos fallo y hacemos fallback
        if len(out) < max(1, int(0.7 * len(candidates))):
            return []
        print("✅ GROQ items devueltos =", len(out), "de", len(candidates))

        return out

    return []


def parse_text_fallback(text: str, allowed: List[str]) -> List[ParsedExpenseItem]:
    """
    Fallback sin IA: usa el mismo pre-parser y mete description raw recortada.
    """
    cands = explode_candidates_from_ocr(text)
    out: List[ParsedExpenseItem] = []
    for c in cands:
        raw = c.get("raw", "").strip()
        # intenta detectar categoría si aparece
        cat = _pick_category_from_text(raw, allowed) or (allowed[0] if allowed else DEFAULT_CATEGORIES[0])
        # limpia raw quitando categoría si está
        if cat:
            raw2 = re.sub(re.escape(cat), "", raw, flags=re.IGNORECASE).strip(" -:\t")
        else:
            raw2 = raw

        desc, extra = _limit_description(raw2 or "Gasto", "")
        out.append(
            ParsedExpenseItem(
                date=c["date"],
                description=desc[:120],
                amount=float(c["amount"]),
                category=cat[:64],
                extra=extra[:120],
                confidence=0.55,
            )
        )
    return out


def get_allowed_categories(user: User, db: Session) -> List[str]:
    custom_rows = (
        db.query(Category)
        .filter(Category.user_id == user.id)
        .order_by(Category.name.asc())
        .all()
    )
    custom = [r.name for r in custom_rows]

    hidden_rows = db.query(HiddenCategory).filter(HiddenCategory.user_id == user.id).all()
    hidden = {r.name.strip().lower() for r in hidden_rows}

    allowed = []
    seen = set()
    for c in DEFAULT_CATEGORIES + custom:
        k = (c or "").strip()
        if not k:
            continue
        if k.lower() in hidden:
            continue
        if k.lower() in seen:
            continue
        seen.add(k.lower())
        allowed.append(k)
    return allowed


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
async def parse_text(
    payload: ParseTextRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    allowed = get_allowed_categories(user, db)

    # 1) pre-parser -> candidatos
    candidates = explode_candidates_from_ocr(payload.text)

    # 2) IA refina (si hay)
    try:
        ai_items = await ai_refine_candidates(candidates, allowed)
        if ai_items:
            return {"items": ai_items}
    except Exception:
        pass

    # 3) fallback sin IA
    items = parse_text_fallback(payload.text, allowed)
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

    text_out, ocr_json = await ocr_via_ocrspace(mem)

    if not text_out.strip():
        return {"items": []}

    allowed = get_allowed_categories(user, db)
    # ✅ 0) Intento robusto con coordenadas (overlay): asocia desc/fecha al importe por posición
    overlay_items = extract_items_from_overlay(ocr_json, allowed)
    if overlay_items:
        return {"items": overlay_items}

    # 1) pre-parser -> candidatos (esto es la CLAVE para que no devuelva 1 solo gasto)
    candidates = explode_candidates_from_ocr(text_out)

    # 2) IA refina candidatos
    try:
        ai_items = await ai_refine_candidates(candidates, allowed)
        if ai_items:
            return {"items": ai_items}
    except Exception:
        pass

    # 3) fallback
    items = parse_text_fallback(text_out, allowed)
    return {"items": items}


@app.get("/")
def root():
    return {"status": "ok", "service": "backend-gastos"}


@app.head("/")
def root_head():
    return Response(status_code=200)
