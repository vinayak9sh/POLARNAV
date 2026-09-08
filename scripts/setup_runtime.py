from pathlib import Path
from urllib.request import urlopen
import sys


# ============================================================
# POLARNAV Runtime Asset Setup
# Release: v1.0.0
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
MODULE1_OUTPUTS_DIR = MODELS_DIR / "module1_outputs"

RELEASE_URL = (
    "https://github.com/vinayak9sh/POLARNAV/releases/download/v1.0.0/"
)

ASSETS = {
    "module1_final_random_forest.joblib":
        MODELS_DIR / "module1_final_random_forest.joblib",

    "module1_final_forecast_runtime.npz":
        MODULE1_OUTPUTS_DIR / "module1_final_forecast_runtime.npz",

    "module2_final_rf_latitude.joblib":
        MODELS_DIR / "module2_final_rf_latitude.joblib",

    "module2_final_rf_longitude.joblib":
        MODELS_DIR / "module2_final_rf_longitude.joblib",
}


def download_file(url: str, destination: Path):
    """Download one runtime asset."""

    print(f"\nDownloading:")
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

            total = response.headers.get("Content-Length")

            if total:
                total = int(total)

            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded * 100 / total
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

        temporary.replace(destination)

        size_mb = destination.stat().st_size / (
            1024 * 1024
        )

        print(
            f"  ✓ Complete ({size_mb:.2f} MB)"
        )

    except Exception as exc:

        if temporary.exists():
            temporary.unlink()

        print(
            f"\n  ✗ Download failed: {exc}"
        )

        raise


def main():

    print("=" * 60)
    print("POLARNAV Runtime Asset Setup")
    print("Release: v1.0.0")
    print("=" * 60)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODULE1_OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename, destination in ASSETS.items():

        url = RELEASE_URL + filename

        if destination.exists():

            size_mb = destination.stat().st_size / (
                1024 * 1024
            )

            print(
                f"\n✓ Already exists: "
                f"{filename} ({size_mb:.2f} MB)"
            )

            continue

        try:
            download_file(
                url,
                destination
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

    print("\nRuntime assets:")

    for _, destination in ASSETS.items():

        size_mb = destination.stat().st_size / (
            1024 * 1024
        )

        print(
            f"  ✓ {destination.relative_to(BASE_DIR)} "
            f"({size_mb:.2f} MB)"
        )


if __name__ == "__main__":
    main()