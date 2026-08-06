"""Stage-1 video quality gate for the web-video -> robot-data pipeline.

Runs on RAW scraped RGB video. Produces a three-way verdict plus the routing
labels that pipeline step 2 consumes. See README.md.
"""
from .config import QCConfig
from .schema import ClipReport, Verdict, ViewClass
from .pipeline import diagnose_clip, diagnose_many

__all__ = ["QCConfig", "ClipReport", "Verdict", "ViewClass",
           "diagnose_clip", "diagnose_many"]
