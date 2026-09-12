# python -m uvicorn src.api:app --reload

from pathlib import Path
from urllib.request import urlopen
import hashlib
import sys


# ============================================================
# POLARNAV Runtime Asset Setup
# Release: v1.1.1
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
MODULE1_OUTPUTS_DIR = MODELS_DIR / "module1_outputs"

RELEASE_URL = (
    "https://github.com/vinayak9sh/POLARNAV/releases/download/v1.1.1/"
)

ASSETS = {
    "module1_final_random_forest.joblib": {
        "path": MODELS_DIR / "module1_final_random_forest.joblib",
        "size": 227886689,
        "sha256": (
            "188EFFD2D9538B1B1C0B8206BCAEF8626D64CBAD361A0B13D22D1CC8934291BC"
        ),
    },

    "module1_final_forecast_runtime.npz": {
        "path": (
            MODULE1_OUTPUTS_DIR
            / "module1_final_forecast_runtime.npz"
        ),
        "size": 23335737,
        "sha256": (
            "08ABC3361D1F86599A00066FD0D014FD7215B924F12A321094EF954C3C9AAEE7"
        ),
    },

    "module1_navigation_runtime.npz": {
    "path": (
        MODULE1_OUTPUTS_DIR
        / "module1_navigation_runtime.npz"
    ),
    "size": 391234,
    "sha256": (
        "FAB0AC62D7AF692AAF3B83B9B609013CDFCC61B2B3FE063B98676117AC76A1B9"
    ),
},

    "module2_final_rf_latitude.joblib": {
        "path": (
            MODELS_DIR
            / "module2_final_rf_latitude.joblib"
        ),
        "size": 65451457,
        "sha256": (
            "55A603D22C6E2ADAA11263A300D3763A4415B0E92483C6FBEC98D3135EB15EC5"
        ),
    },

    "module2_final_rf_longitude.joblib": {
        "path": (
            MODELS_DIR
            / "module2_final_rf_longitude.joblib"
        ),
        "size": 63695233,
        "sha256": (
            "5A215BDF995578E2FC2084CA5A65AB7C96B4069BCCF0208449F824E47D3329F8"
        ),
    },

    "nsidc_monthly_2018_2025_polarnav.nc": {
        "path": (
            BASE_DIR
            / "data"
            / "processed"
            / "nsidc_monthly_2018_2025_polarnav.nc"
        ),
        "size": 71678754,
        "sha256": (
            "9ED40B22FD91A6A1C7DD1F903AECBBE297B04085A7C8C5972C54AFCAA2E1A859"
        ),
    },
}

def calculate_sha256(path: Path):
    """Calculate SHA-256 for a file."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()

def verify_asset(
    path: Path,
    expected_size: int,
    expected_sha256: str,
):
    """
    Verify an existing runtime asset using
    exact byte size and SHA-256.
    """

    if not path.exists():
        return False

    actual_size = path.stat().st_size

    if actual_size != expected_size:
        print(
            f"\n  ✗ Invalid size: {path.name}"
        )
        print(
            f"    Expected: {expected_size:,} bytes"
        )
        print(
            f"    Found:    {actual_size:,} bytes"
        )
        return False

    actual_sha256 = calculate_sha256(
        path
    )

    if actual_sha256.lower() != expected_sha256.lower():
        print(
            f"\n  ✗ Invalid SHA-256: {path.name}"
        )
        print(
            f"    Expected: {expected_sha256}"
        )
        print(
            f"    Found:    {actual_sha256}"
        )
        return False

    return True

def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
):
    """Download one runtime asset and verify its integrity."""

    print("\nDownloading:")
    print(f"  {destination.name}")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = destination.with_suffix(
        destination.suffix + ".download"
    )

    try:
        with urlopen(url) as response, open(
            temporary,
            "wb"
        ) as output:

            total = response.headers.get(
                "Content-Length"
            )

            if total:
                total = int(total)

            downloaded = 0

            while True:
                chunk = response.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = (
                        downloaded * 100 / total
                    )

                    print(
                        f"\r  Progress: {percent:6.2f}%",
                        end=""
                    )
                else:
                    print(
                        f"\r  Downloaded: "
                        f"{downloaded / 1024 / 1024:.1f} MB",
                        end=""
                    )

        print()

        # --------------------------------------------------
        # Verify downloaded file BEFORE replacing the
        # destination.
        # --------------------------------------------------

        actual_size = temporary.stat().st_size

        print(
            f"  Checking size: "
            f"{actual_size:,} / "
            f"{expected_size:,} bytes"
        )

        if actual_size != expected_size:
            raise ValueError(
                "Downloaded file size does not match "
                "the expected size."
            )

        actual_sha256 = calculate_sha256(
            temporary
        )

        print(
            f"  SHA-256: "
            f"{actual_sha256}"
        )

        if actual_sha256.lower() != (
            expected_sha256.lower()
        ):
            raise ValueError(
                "Downloaded file SHA-256 does not "
                "match the expected hash."
            )

        # --------------------------------------------------
        # Only install the file after successful
        # verification.
        # --------------------------------------------------

        temporary.replace(
            destination
        )

        size_mb = (
            destination.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"  ✓ Verified and installed "
            f"({size_mb:.2f} MB)"
        )

    except Exception as exc:

        if temporary.exists():
            temporary.unlink()

        print(
            f"\n  ✗ Download failed or failed "
            f"integrity verification: {exc}"
        )

        raise

def main():

    print("=" * 60)
    print("POLARNAV Runtime Asset Setup")
    print("Release: v1.1.1")
    print("=" * 60)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODULE1_OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename, info in ASSETS.items():

        destination = info["path"]
        expected_size = info["size"]
        expected_sha256 = info["sha256"]

        # --------------------------------------------------
        # Existing file: verify before trusting it.
        # --------------------------------------------------

        if destination.exists():

            print(
                f"\nChecking existing asset: "
                f"{filename}"
            )

            if verify_asset(
                destination,
                expected_size,
                expected_sha256,
            ):
                size_mb = (
                    destination.stat().st_size
                    / (1024 * 1024)
                )

                print(
                    f"  ✓ Verified "
                    f"({size_mb:.2f} MB)"
                )

                continue

            print(
                "  → Existing file is invalid."
            )
            print(
                "  → Removing and redownloading..."
            )

            destination.unlink()

        # --------------------------------------------------
        # Download missing or invalid asset.
        # --------------------------------------------------

        url = RELEASE_URL + filename

        try:
            download_file(
                url=url,
                destination=destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )

        except Exception:

            print(
                "\nRuntime setup failed."
            )

            print(
                "Check your internet connection "
                "and GitHub release assets."
            )

            sys.exit(1)

    print("\n" + "=" * 60)
    print("Runtime setup completed successfully.")
    print("=" * 60)

    print("\nVerified runtime assets:")

    for filename, info in ASSETS.items():

        destination = info["path"]

        size_mb = (
            destination.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"  ✓ "
            f"{destination.relative_to(BASE_DIR)} "
            f"({size_mb:.2f} MB)"
        )

if __name__ == "__main__":
    main()