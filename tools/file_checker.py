from pathlib import Path
import gzip
import zipfile
import shutil

current_file_path = Path(__file__).resolve()
current_dir       = current_file_path.parent.parent

WEATHER_DATA_DIR  = current_dir / "data" / "weather_data"

def detect_file_type(path):
    sig = b""
    try:
        with path.open("rb") as f:
            sig = f.read(8)
    except Exception:
        return "unreadable"

    if sig.startswith(b"\x89HDF\r\n\x1a\n"):
        return "netCDF4 (HDF5)"
    if sig.startswith(b"CDF"):
        return "netCDF classic"
    if sig.startswith(b"\x1f\x8b"):
        # gzip: inspect inner bytes
        try:
            with gzip.open(path, "rb") as gz:
                inner = gz.read(8)
            if inner.startswith(b"\x89HDF\r\n\x1a\n"):
                return "gzip compressed netCDF4 (HDF5)"
            if inner.startswith(b"CDF"):
                return "gzip compressed netCDF classic"
            return "gzip compressed (not netCDF)"
        except Exception:
            return "gzip compressed (could not inspect)"
    if sig.startswith(b"PK\x03\x04"):
        # zip archive: inspect members
        try:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    try:
                        with z.open(name) as member:
                            inner = member.read(8)
                        if inner.startswith(b"\x89HDF\r\n\x1a\n"):
                            return f"zip containing netCDF4 (member: {name})"
                        if inner.startswith(b"CDF"):
                            return f"zip containing netCDF classic (member: {name})"
                    except Exception:
                        continue
                return "zip archive (no netCDF detected)"
        except Exception:
            return "zip archive (could not inspect)"
    return "unknown"

# iterate files and report
if WEATHER_DATA_DIR.exists():
    for p in WEATHER_DATA_DIR.iterdir():
        if p.is_file():
            t = detect_file_type(p)
            print(p.name, "->", t)
            if t.startswith("zip containing netCDF"):
                tempName = p.name
                try:
                    # ensure .zip extension (avoid collisions)
                    zip_path = p
                    if zip_path.suffix.lower() != ".zip":
                        candidate = zip_path.with_suffix(".zip")
                        idx = 1
                        while candidate.exists():
                            candidate = zip_path.with_name(f"{zip_path.stem}_{idx}.zip")
                            idx += 1
                        zip_path = zip_path.rename(candidate)

                    # extract to a unique temp dir inside WEATHER_DATA_DIR
                    temp_dir = WEATHER_DATA_DIR / f"{zip_path.stem}_unzipped"
                    idx = 1
                    while temp_dir.exists():
                        temp_dir = WEATHER_DATA_DIR / f"{zip_path.stem}_unzipped_{idx}"
                        idx += 1
                    temp_dir.mkdir(parents=True)

                    with zipfile.ZipFile(zip_path) as z:
                        z.extractall(temp_dir)

                    # move all extracted contents into WEATHER_DATA_DIR (resolve name collisions)
                    for item in temp_dir.iterdir():
                        dest = WEATHER_DATA_DIR / item.name
                        if dest.exists():
                            i = 1
                            base = item.stem
                            suf = item.suffix
                            new_dest = WEATHER_DATA_DIR / f"{base}_{i}{suf}"
                            while new_dest.exists():
                                i += 1
                                new_dest = WEATHER_DATA_DIR / f"{base}_{i}{suf}"
                            dest = new_dest
                        shutil.move(str(item), str(dest))

                    # remove the now-empty temp dir and the zip file
                    try:
                        temp_dir.rmdir()
                    except Exception:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    zip_path.unlink()
                    temp_zip_name = zip_path.name
                    print("Processed and removed zip:", zip_path.name)

                    # rename an associated .nc file to match the zip base name (keep .nc extension)
                    target_stem = Path(temp_zip_name).stem
                    target_path = WEATHER_DATA_DIR / f"{target_stem}.nc"
                    for nc_file in WEATHER_DATA_DIR.glob("*.nc"):
                        if nc_file.name == target_path.name:
                            continue
                        dest = target_path
                        i = 1
                        while dest.exists():
                            dest = WEATHER_DATA_DIR / f"{target_stem}_{i}.nc"
                            i += 1
                        try:
                            shutil.move(str(nc_file), str(dest))
                            print("Renamed", nc_file.name, "->", dest.name)
                        except Exception as e:
                            print("Failed renaming", nc_file.name, "->", dest.name, ":", e)
                        break
                except Exception as e:
                    print("Failed processing", p.name, "->", e)
else:
    print("WEATHER_DATA_DIR does not exist:", WEATHER_DATA_DIR)

