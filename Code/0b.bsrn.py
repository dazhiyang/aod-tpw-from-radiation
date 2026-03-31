"""BSRN FTP: inventory for ``YEAR_TO_CHECK``, then download 12× monthly ``cccmmyy.dat.gz`` per station.

Writes under ``Data/BSRN/PAL``, ``Data/BSRN/QIQ``, ``Data/BSRN/TAT``. Credentials: ``BSRN_USER``,
``BSRN_PASS`` (optional defaults for public read-only FTP if your site uses them).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import bsrn

# Calendar year used for the per-station availability line and for which monthlies are downloaded.
YEAR_TO_CHECK = 2024

PROJECT = Path(__file__).resolve().parent.parent
DATA_BSRN = PROJECT / "Data" / "BSRN"

username = os.environ.get("BSRN_USER", "bsrnftp")
password = os.environ.get("BSRN_PASS", "bsrn1")

STATIONS = ("PAL", "QIQ", "TAT")

_RE = re.compile(r"^[a-z]{3}(\d{2})(\d{2})\.dat\.gz$", re.IGNORECASE)


def _year_from_name(name: str) -> int | None:
    m = _RE.match(name)
    if not m:
        return None
    yy = int(m.group(2))
    return 1900 + yy if yy >= 50 else 2000 + yy


inventory = bsrn.io.retrieval.get_bsrn_file_inventory(list(STATIONS), username, password)

yy = YEAR_TO_CHECK % 100
print(f"Year {YEAR_TO_CHECK} (yy={yy:02d}) — LR0100 monthly count on server / station")
for stn in STATIONS:
    files = inventory.get(stn, [])
    n_y = sum(1 for f in files if _year_from_name(f) == YEAR_TO_CHECK)
    print(f"  {stn}:  {n_y} / {len(files)}  (this year / all names)")

print()
for stn in STATIONS:
    out = DATA_BSRN / stn
    out.mkdir(parents=True, exist_ok=True)
    prefix = stn.lower()
    names = [f"{prefix}{m:02d}{yy:02d}.dat.gz" for m in range(1, 13)]
    print(f"Download {stn} -> {out}")
    paths = bsrn.io.retrieval.download_bsrn_files(
        filenames=names,
        local_dir=str(out.resolve()),
        username=username,
        password=password,
    )
    ok = sum(1 for p in paths if p)
    print(f"  {ok}/{len(names)} ok")
