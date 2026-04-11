from pathlib import Path
from dotenv import load_dotenv

from utils import detect_device

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    ROOT_DIR = Path("../").resolve()
    DEVICE = detect_device()

    """
    Pipeline settings:
    """
