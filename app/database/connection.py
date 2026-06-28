import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

# Configure SQLAlchemy Engine
DATABASE_URL = settings.DATABASE_URL
connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    # SQLite requires check_same_thread=False for multi-threaded FastAPI
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error("Failed to initialize database engine for url %s: %s", DATABASE_URL, e)
    # Fallback to local sqlite to ensure the app doesn't crash on start
    fallback_url = "sqlite:///vectorstores/sessions.db"
    logger.info("Falling back to local SQLite database: %s", fallback_url)
    engine = create_engine(fallback_url, connect_args={"check_same_thread": False}, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database and create all tables."""
    try:
        # Create output directory for database if it's sqlite
        if settings.DATABASE_URL.startswith("sqlite"):
            from pathlib import Path
            db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize database tables: %s", e)


def get_db():
    """Dependency for API endpoints or RAG pipelines to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
