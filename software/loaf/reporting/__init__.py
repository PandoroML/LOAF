"""Standalone HTML training reports for LOAF runs.

See loaf.reporting.collect.build_report_data() for assembling a single run's
report data and loaf.reporting.collect.build_index_data() for a cross-run
"master report" summarizing everything under a runs/ directory;
loaf.reporting.render has the matching render_html()/render_index_html().
scripts/report.py (`loaf-report`) and scripts/report_summary.py
(`loaf-report-summary`) are the CLI entry points.
"""

from loaf.reporting.collect import (
    IndexData,
    ReportData,
    RunSummary,
    build_index_data,
    build_report_data,
    discover_runs,
)
from loaf.reporting.render import render_html, render_index_html

__all__ = [
    "IndexData",
    "ReportData",
    "RunSummary",
    "build_index_data",
    "build_report_data",
    "discover_runs",
    "render_html",
    "render_index_html",
]
