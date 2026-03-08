"""
organize_data.py — Transforms your raw downloaded data into
the exact structure the pipeline expects.

WHAT IT FIXES:
  1. Renames eye_lids → conjunctiva
  2. Flattens Anemic/Non-anemic subfolders into one original/ folder
  3. Adds .png extension to Excel IMAGE_ID entries
  4. Filters out augmented images (filenames containing parentheses)
  5. Generates labels.csv for each modality

YOUR CURRENT STRUCTURE:
    data/
    ├── eye_lids/
    │   ├── Anemic/
    │   │   ├── Image_001.png
    │   │   └── ...
    │   ├── Non-anemic/
    │   │   ├── Image_050.png
    │   │   └── ...
    │   └── Anemia_Data_Collection_Sheet.xlsx
    ├── nail/
    │   ├── Anemic-Fin-001.png
    │   ├── Anemic-Fin-007 (10).png       ← AUGMENTED (will be skipped)
    │   ├── Non-Anemic-Fin-001.png
    │   └── ...
    └── palm/
        ├── Anemic-Plm-001.png
        ├── Anemic-Plm-007 (5).png        ← AUGMENTED (will be skipped)
        ├── Non-Anemic-Plm-001.png
        └── ...

AFTER RUNNING THIS SCRIPT:
    data/
    ├── conjunctiva/
    │   ├── original/
    │   │   ├── Image_001.png
    │   │   ├── Image_002.png
    │   │   └── ...
    │   └── labels.csv          ← real Hb values from Excel
    ├── nail/
    │   ├── original/
    │   │   ├── Anemic-Fin-001.png        ← only originals
    │   │   ├── Non-Anemic-Fin-001.png
    │   │   └── ...
    │   └── labels.csv          ← dummy Hb (9.0 anemic / 13.0 healthy)
    └── palm/
        ├── original/
        │   ├── Anemic-Plm-001.png        ← only originals
        │   ├── Non-Anemic-Plm-001.png
        │   └── ...
        └── labels.csv          ← dummy Hb (9.0 anemic / 13.0 healthy)
"""

import os
import shutil
import pandas as pd
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

TARGETS = {
    "conjunctiva": DATA_DIR / "conjunctiva",
    "nail":        DATA_DIR / "nail",
    "palm":        DATA_DIR / "palm",
}


# ── Step 1: Create folder structure ───────────────────────────
def setup_folders():
    print("\n" + "=" * 55)
    print("  STEP 1: Creating folder structure")
    print("=" * 55)
    for name, path in TARGETS.items():
        original_dir = path / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {original_dir}")


# ── Step 2: Process Eyes (conjunctiva) ────────────────────────
def process_eyes():
    print("\n" + "=" * 55)
    print("  STEP 2: Processing Eyes → conjunctiva")
    print("=" * 55)

    eye_source = DATA_DIR / "eye_lids"

    if not eye_source.exists():
        print(f"  ❌ Cannot find folder: {eye_source}")
        print(f"     Make sure your eye images are in data/eye_lids/")
        return False

    # ── Find the Excel file ──────────────────────────────────
    excel_path = eye_source / "Anemia_Data_Collection_Sheet.xlsx"

    if not excel_path.exists():
        # Try to find any xlsx file in the folder
        xlsx_files = list(eye_source.glob("*.xlsx"))
        if xlsx_files:
            excel_path = xlsx_files[0]
            print(f"  ⚠  Using found Excel: {excel_path.name}")
        else:
            print(f"  ❌ No Excel file found in {eye_source}")
            print(f"     Expected: Anemia_Data_Collection_Sheet.xlsx")
            return False

    print(f"  Reading: {excel_path.name}")
    df = pd.read_excel(excel_path)
    print(f"  Excel columns: {list(df.columns)}")
    print(f"  Excel rows: {len(df)}")

    # ── Identify columns ─────────────────────────────────────
    #    Handle various possible column names
    image_col = None
    hb_col = None

    for col in df.columns:
        col_lower = str(col).strip().lower().replace(" ", "_")
        if "image" in col_lower or "img" in col_lower:
            image_col = col
        if "hb" in col_lower or "hemoglobin" in col_lower or "haemoglobin" in col_lower:
            hb_col = col

    if image_col is None or hb_col is None:
        print(f"  ❌ Cannot identify columns automatically.")
        print(f"     Found: {list(df.columns)}")
        print(f"     Need one column with 'image' and one with 'hb'")
        return False

    print(f"  Image column: '{image_col}'")
    print(f"  Hb column:    '{hb_col}'")

    # ── Process each row ─────────────────────────────────────
    csv_data = []
    copied = 0
    skipped = 0

    for _, row in df.iterrows():
        img_base = str(row[image_col]).strip()
        hb_level = row[hb_col]

        # Skip rows with missing data
        if pd.isna(hb_level) or img_base == "nan":
            skipped += 1
            continue

        hb_level = float(hb_level)

        # Add .png extension if missing
        if not img_base.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_name = f"{img_base}.png"
        else:
            img_name = img_base

        # Search in both Anemic and Non-anemic subfolders
        src_path = None
        for subfolder in ("Anemic", "Non-anemic", "anemic",
                          "non-anemic", "Non-Anemic", "NonAnemic"):
            candidate = eye_source / subfolder / img_name
            if candidate.exists():
                src_path = candidate
                break

        # Also check the root eye_lids folder
        if src_path is None:
            candidate = eye_source / img_name
            if candidate.exists():
                src_path = candidate

        if src_path is not None:
            dest_path = TARGETS["conjunctiva"] / "original" / img_name
            shutil.copy2(src_path, dest_path)
            csv_data.append({
                "image_filename": img_name,
                "hb_level": hb_level,
            })
            copied += 1
        else:
            print(f"    ⚠  Not found: {img_name}")
            skipped += 1

    # ── Save labels.csv ──────────────────────────────────────
    if csv_data:
        labels_df = pd.DataFrame(csv_data)
        csv_path = TARGETS["conjunctiva"] / "labels.csv"
        labels_df.to_csv(csv_path, index=False)

        anemic = (labels_df["hb_level"] < 11.0).sum()
        healthy = (labels_df["hb_level"] >= 11.0).sum()

        print(f"\n  ✅ Conjunctiva complete!")
        print(f"     Images copied  : {copied}")
        print(f"     Images skipped : {skipped}")
        print(f"     Hb range       : {labels_df['hb_level'].min():.1f} – "
              f"{labels_df['hb_level'].max():.1f} g/dL")
        print(f"     Anemic (Hb<11) : {anemic}")
        print(f"     Healthy (Hb≥11): {healthy}")
        print(f"     Labels saved   : {csv_path}")
    else:
        print(f"  ❌ No images were processed!")
        return False

    return True


# ── Step 3: Process Nails / Palms ─────────────────────────────
def process_binary_modality(modality_name, source_folder_name):
    print(f"\n{'=' * 55}")
    print(f"  STEP 3: Processing {modality_name.upper()}")
    print(f"{'=' * 55}")

    source_dir = DATA_DIR / source_folder_name

    if not source_dir.exists():
        print(f"  ❌ Cannot find folder: {source_dir}")
        return False

    # ── Count what we have before filtering ──────────────────
    all_images = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    print(f"  Total files found: {len(all_images)}")

    csv_data = []
    copied = 0
    augmented_skipped = 0

    for file_path in sorted(all_images):
        filename = file_path.name

        # ── CRITICAL: Skip augmented images ──────────────────
        #    Augmented files contain parentheses: "Anemic-Fin-007 (10).png"
        #    Original files do NOT: "Anemic-Fin-007.png"
        if "(" in filename or ")" in filename:
            augmented_skipped += 1
            continue

        # ── Determine class from filename ────────────────────
        name_lower = filename.lower()

        if "non" in name_lower:
            hb_level = 13.0      # healthy
        elif "anemic" in name_lower or "anaemic" in name_lower:
            hb_level = 9.0       # anemic
        else:
            # Cannot determine class — skip with warning
            print(f"    ⚠  Cannot classify: {filename} (skipping)")
            continue

        # ── Copy to original/ folder ─────────────────────────
        dest_path = TARGETS[modality_name] / "original" / filename
        shutil.copy2(file_path, dest_path)

        csv_data.append({
            "image_filename": filename,
            "hb_level": hb_level,
        })
        copied += 1

    # ── Save labels.csv ──────────────────────────────────────
    if csv_data:
        labels_df = pd.DataFrame(csv_data)
        csv_path = TARGETS[modality_name] / "labels.csv"
        labels_df.to_csv(csv_path, index=False)

        anemic = (labels_df["hb_level"] < 11.0).sum()
        healthy = (labels_df["hb_level"] >= 11.0).sum()

        print(f"\n  ✅ {modality_name.capitalize()} complete!")
        print(f"     Original images copied  : {copied}")
        print(f"     Augmented images SKIPPED : {augmented_skipped}")
        print(f"     Anemic  (Hb=9.0)  : {anemic}")
        print(f"     Healthy (Hb=13.0) : {healthy}")
        print(f"     Labels saved      : {csv_path}")

        if anemic == 0:
            print(f"\n  ⚠️  WARNING: Zero anemic images!")
            print(f"     Model cannot learn without both classes.")
        if healthy == 0:
            print(f"\n  ⚠️  WARNING: Zero healthy images!")
            print(f"     Model cannot learn without both classes.")
    else:
        print(f"  ❌ No original images found!")
        return False

    return True


# ── Summary ───────────────────────────────────────────────────
def print_summary():
    print(f"\n{'=' * 55}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 55}")

    for name, path in TARGETS.items():
        original_dir = path / "original"
        csv_path = path / "labels.csv"

        if original_dir.exists():
            n_images = len(list(original_dir.glob("*.*")))
        else:
            n_images = 0

        csv_ok = csv_path.exists()

        status = "✅" if (n_images > 0 and csv_ok) else "❌"
        print(f"  {status}  {name:15s}  {n_images:4d} images  "
              f"csv={'YES' if csv_ok else 'NO'}")

    print(f"\n  Your data/ folder now looks like:")
    print(f"  data/")
    for name in ("conjunctiva", "nail", "palm"):
        path = TARGETS[name]
        original_dir = path / "original"
        n = len(list(original_dir.glob("*.*"))) if original_dir.exists() else 0
        print(f"  ├── {name}/")
        print(f"  │   ├── original/    ({n} images)")
        print(f"  │   └── labels.csv")
    print()


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DATA ORGANIZER")
    print("  Transforms raw downloads → pipeline-ready structure")
    print("=" * 55)

    setup_folders()

    eye_ok  = process_eyes()
    nail_ok = process_binary_modality("nail", "nail")
    palm_ok = process_binary_modality("palm", "palm")

    print_summary()

    if eye_ok and nail_ok and palm_ok:
        print("  🎉 ALL DONE! Next steps:")
        print("     1. python verify_setup.py")
        print("     2. python main.py")
    else:
        print("  ⚠️  Some modalities had issues. Fix them before training.")
