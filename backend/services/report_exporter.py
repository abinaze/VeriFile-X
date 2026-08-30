"""
Forensic Report Export Suite.

Exports forensic analysis reports in multiple formats:

  PDF  - Court-ready forensic report with metadata, hashes,
         detection signals table, and executive summary.
         Uses only stdlib + Pillow — no external PDF library required.
         Generates a minimal but complete PDF/1.4 document.

  JSON - Full structured export, identical to API response.

  CSV  - Signal-level flat export for spreadsheet analysis.
         One row per detection signal with all fields.

All exports include:
  - Evidence ID (UUID)
  - Analysis timestamp
  - File hash (SHA-256 and MD5)
  - AI probability and classification
  - Per-signal breakdown (26 signals)
  - Generator attribution
  - Platform forensics
  - C2PA provenance status

Reached only via POST /api/v1/analyze/export/{fmt}, which re-analyzes the
uploaded image from scratch — this module never receives an already-computed
report. The frontend's own "Download PDF Report" button does NOT call this
endpoint (H-5, part B): it renders a PDF client-side, instantly, from the
report and thumbnail already in the browser's memory, specifically to avoid
a second upload plus a second full analysis pass just to get a PDF of a
report the user is already looking at. This module's PDF output has no
thumbnail and is meant as a plain, re-derivable evidentiary artifact for a
direct API consumer, not as a replacement for that button. If you change the
shape of one, check whether the other should change too, but they are
deliberately not the same code path.
"""
import json
import csv
import io
from backend.core.logger import setup_logger
import zlib
from typing import Dict, Any

logger = setup_logger(__name__)

# Characters that Excel/LibreOffice Calc/Google Sheets treat as the start
# of a formula when a CSV cell is opened in a spreadsheet (CWE-1236).
_CSV_FORMULA_CHARS = ("=", "+", "-", "@")


def _csv_safe(value: Any) -> str:
    """Neutralize CSV/formula injection (F-2).

    Any cell whose first character is one of _CSV_FORMULA_CHARS is
    prefixed with a leading apostrophe, which spreadsheet applications
    treat as "force this cell to plain text" rather than a formula.
    Filenames and EXIF-derived text (e.g. metadata_forensics.py's
    "Software: <exif value>" explanation) are attacker-influenceable and
    flow into this export unescaped otherwise.
    """
    s = str(value)
    if s and s[0] in _CSV_FORMULA_CHARS:
        return "'" + s
    return s


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(report: Dict[str, Any], indent: int = 2) -> bytes:
    """Export full forensic report as formatted JSON bytes."""
    return json.dumps(report, indent=indent, default=str).encode("utf-8")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(report: Dict[str, Any]) -> bytes:
    """
    Export signal-level data as CSV.

    Columns: evidence_id, filename, analysis_timestamp, ai_probability,
             classification, signal_name, score, confidence, explanation,
             raw_value, expected_range
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    evidence_id = report.get("evidence_id", "")
    filename    = report.get("file_info", {}).get("filename", "")
    timestamp   = report.get("metadata", {}).get("analysis_timestamp", "")
    ai_prob     = report.get("summary", {}).get("ai_probability", 0.0)
    ai_class    = report.get("summary", {}).get("ai_classification", "")

    writer.writerow([
        "evidence_id", "filename", "analysis_timestamp",
        "ai_probability", "classification",
        "signal_name", "score", "confidence",
        "explanation", "raw_value", "expected_range",
    ])

    signals = report.get("ai_detection", {}).get("all_signals", [])
    if not signals:
        writer.writerow([
            _csv_safe(evidence_id), _csv_safe(filename), _csv_safe(timestamp),
            ai_prob, _csv_safe(ai_class),
            "no_signals", "", "", "", "", "",
        ])
    else:
        for sig in signals:
            writer.writerow([
                _csv_safe(evidence_id),
                _csv_safe(filename),
                _csv_safe(timestamp),
                ai_prob,
                _csv_safe(ai_class),
                _csv_safe(sig.get("signal_name", "")),
                sig.get("score", ""),
                sig.get("confidence", ""),
                _csv_safe(sig.get("explanation", "").replace("\n", " ")),
                sig.get("raw_value", ""),
                _csv_safe(sig.get("expected_range", "")),
            ])

    return buf.getvalue().encode("utf-8")


# ── PDF export ────────────────────────────────────────────────────────────────
# Minimal PDF/1.4 built from scratch using only stdlib.
# No external PDF library required — CI compatible.

def _pdf_stream(content: bytes) -> tuple:
    """Compress content with zlib, return (compressed_bytes, length)."""
    compressed = zlib.compress(content)
    return compressed, len(compressed)


class _PDFWriter:
    """
    Minimal PDF/1.4 writer — no dependencies.

    build() (the only method ever called on instances of this class) uses
    zero instance state — verified empirically (0 "self." references
    anywhere in build(), self._buf confirmed untouched at its initial 15
    bytes after calling .build()). __init__ previously set up
    self._objects/_offsets/_buf plus three helper methods
    (_add_object/_write_object/_text_stream) purely for build() to ignore
    entirely, since build() re-derives everything with fresh local
    variables instead. Removed as dead code.
    """

    def __init__(self):
        pass

    def build(self, report: Dict[str, Any]) -> bytes:
        summary  = report.get("summary", {})
        file_info = report.get("file_info", {})
        hashes   = report.get("hashes", {})
        ai_det   = report.get("ai_detection", {})
        attr     = report.get("generator_attribution", {})
        platform = report.get("platform_forensics", {})
        c2pa     = report.get("c2pa_provenance", {})

        evidence_id = report.get("evidence_id", "N/A")
        filename    = file_info.get("filename", "N/A")
        timestamp   = report.get("metadata", {}).get("analysis_timestamp", "N/A")
        ai_prob     = summary.get("ai_probability", 0.0)
        ai_class    = summary.get("ai_classification", "N/A")
        sha256      = hashes.get("sha256", "N/A")
        md5         = hashes.get("md5", "N/A")
        generator   = attr.get("predicted_generator", "N/A")
        platform_o  = platform.get("predicted_platform", "N/A")
        c2pa_status = c2pa.get("provenance_status", "N/A")

        # Build page 1 content stream
        text_ops = []
        text_ops.append(b"BT")
        text_ops.append(b"/F1 16 Tf")
        text_ops.append(b"50 780 Td")
        text_ops.append(b"(VeriFile-X Forensic Analysis Report) Tj")
        text_ops.append(b"/F1 9 Tf")
        text_ops.append(b"0 -20 Td")
        text_ops.append(f"(Generated: {timestamp[:19]}) Tj".encode("latin-1", errors="replace"))

        def row(label, value, dy=-14):
            safe_v = str(value).replace("(", "[").replace(")", "]").replace("\\", "/")
            return [
                f"0 {dy} Td".encode(),
                f"({label}: {safe_v[:80]}) Tj".encode(),
            ]

        text_ops.append(b"0 -22 Td")
        text_ops.append(b"/F1 11 Tf")
        text_ops.append(b"(EVIDENCE INFORMATION) Tj")
        text_ops.append(b"/F1 9 Tf")
        for label, val in [
            ("Evidence ID",      evidence_id),
            ("Filename",         filename),
            ("SHA-256",          sha256[:32] + "..."),
            ("MD5",              md5),
            ("Width x Height",   f"{file_info.get('width','?')}x{file_info.get('height','?')} px"),
            ("File size",        f"{file_info.get('file_size_bytes', 0):,} bytes"),
        ]:
            for op in row(label, val):
                text_ops.append(op)

        text_ops.append(b"0 -18 Td")
        text_ops.append(b"/F1 11 Tf")
        text_ops.append(b"(AI DETECTION RESULT) Tj")
        text_ops.append(b"/F1 9 Tf")
        prob_pct = f"{ai_prob * 100:.1f}%"
        for label, val in [
            ("AI Probability",      prob_pct),
            ("Classification",      ai_class),
            ("Total Signals",       summary.get("total_detection_signals", "N/A")),
            ("Suspicious Signals",  summary.get("suspicious_detection_signals", "N/A")),
            ("Generator",           generator),
            ("Platform Origin",     platform_o),
            ("C2PA Status",         c2pa_status),
        ]:
            for op in row(label, val):
                text_ops.append(op)

        text_ops.append(b"0 -18 Td")
        text_ops.append(b"/F1 11 Tf")
        text_ops.append(b"(DETECTION SIGNALS) Tj")
        text_ops.append(b"/F1 8 Tf")
        signals = ai_det.get("all_signals", [])
        for sig in signals[:20]:
            name  = sig.get("signal_name", "")[:35]
            score = f"{sig.get('score', 0):.3f}"
            conf  = f"{sig.get('confidence', 0):.2f}"
            flag  = "SUSPICIOUS" if sig.get("score", 0) > 0.5 else "ok"
            line  = f"{name:<35} score={score}  conf={conf}  [{flag}]"
            safe  = line.replace("(", "[").replace(")", "]").replace("\\", "/")
            text_ops.append(f"0 -11 Td ({safe}) Tj".encode("latin-1", errors="replace"))

        if len(signals) > 20:
            text_ops.append(f"0 -11 Td ({len(signals) - 20} additional signals in JSON export.) Tj".encode())

        text_ops.append(b"0 -18 Td")
        text_ops.append(b"/F1 11 Tf")
        text_ops.append(b"(LEGAL NOTICE) Tj")
        text_ops.append(b"/F1 8 Tf")
        text_ops.append(b"0 -13 Td (This report was generated by VeriFile-X automated forensic analysis.) Tj")
        text_ops.append(b"0 -11 Td (Results are probabilistic and should be reviewed by a qualified forensic expert.) Tj")
        text_ops.append(b"0 -11 Td (This report does not constitute legal evidence without expert verification.) Tj")
        text_ops.append(b"ET")

        page_content = b"\n".join(text_ops)
        comp, comp_len = _pdf_stream(page_content)

        # Object 1: Catalog
        # Object 2: Pages
        # Object 3: Page
        # Object 4: Font
        # Object 5: Content stream

        buf = io.BytesIO()
        buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}

        def write_obj(num, body):
            offsets[num] = buf.tell()
            buf.write(f"{num} 0 obj\n".encode())
            buf.write(body)
            buf.write(b"\nendobj\n")

        write_obj(4, (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"
        ))
        write_obj(5, (
            f"<< /Filter /FlateDecode /Length {comp_len} >>\nstream\n".encode() +
            comp + b"\nendstream"
        ))
        write_obj(3, (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 842] "
            b"/Contents 5 0 R "
            b"/Resources << /Font << /F1 4 0 R >> >> >>"
        ))
        write_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")

        xref_offset = buf.tell()
        buf.write(b"xref\n")
        buf.write(b"0 6\n")
        buf.write(b"0000000000 65535 f \n")
        for i in range(1, 6):
            buf.write(f"{offsets[i]:010d} 00000 n \n".encode())

        buf.write(
            f"trailer\n<< /Size 6 /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode()
        )
        return buf.getvalue()


def export_pdf(report: Dict[str, Any]) -> bytes:
    """Export forensic report as a minimal PDF/1.4 document."""
    try:
        writer = _PDFWriter()
        return writer.build(report)
    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        # Return a minimal error PDF
        return _error_pdf(str(e))


def _error_pdf(message: str) -> bytes:
    """Return a minimal PDF with error message."""
    content  = f"BT /F1 12 Tf 50 700 Td (PDF generation error: {message[:80]}) Tj ET".encode("latin-1", errors="replace")
    comp, cl = _pdf_stream(content)
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets  = {}

    def wo(n, b):
        offsets[n] = buf.tell()
        buf.write(f"{n} 0 obj\n".encode())
        buf.write(b)
        buf.write(b"\nendobj\n")

    wo(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    wo(5, (f"<< /Filter /FlateDecode /Length {cl} >>\nstream\n".encode() + comp + b"\nendstream"))
    wo(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>")
    wo(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    wo(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    xoff = buf.tell()
    buf.write(b"xref\n0 6\n0000000000 65535 f \n")
    for i in range(1, 6):
        buf.write(f"{offsets[i]:010d} 00000 n \n".encode())
    buf.write(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xoff}\n%%EOF\n".encode())
    return buf.getvalue()
