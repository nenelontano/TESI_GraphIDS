import zipfile
from pathlib import Path

ESCLUSI = ("predictions", "count_baseline", "split_flows")  # file pesanti che all'analisi non servono

with zipfile.ZipFile("risultati_leggeri.zip", "w", zipfile.ZIP_DEFLATED) as z:
    n = 0
    for p in Path("data").glob("output_*/**/*"):
        if p.is_file() and (p.name == "run.json" or p.suffix == ".csv") and not p.name.startswith(ESCLUSI):
            z.write(p)
            n += 1

print(f"risultati_leggeri.zip: {n} file, {Path('risultati_leggeri.zip').stat().st_size / 1e6:.2f} MB")