import os
import re
import zipfile
import tarfile
import gzip
import shutil
import requests


def extract_record_id(url_or_id):
    """Extracts the numerical record ID from a Zenodo URL or identifier string."""
    match = re.search(r"record[s]?/(\d+)", str(url_or_id))
    if match:
        return match.group(1)
    if str(url_or_id).isdigit():
        return str(url_or_id)
    raise ValueError(f"Could not parse Zenodo record ID from: {url_or_id}")


def extract_archive(file_path, output_dir):
    """Detects and extracts common compression formats using built-in libraries."""
    filename = os.path.basename(file_path).lower()

    # 1. Zip archives
    if filename.endswith(".zip"):
        print(f"📦 Extracting zip archive: {os.path.basename(file_path)}...")
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            print("✅ Extraction complete.")
        except Exception as e:
            print(f"❌ Failed to extract zip file: {e}")

    # 2. Tarball archives (.tar, .tar.gz, .tgz, .tar.bz2, etc.)
    elif filename.endswith(
        (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    ):
        print(f"📦 Extracting tar archive: {os.path.basename(file_path)}...")
        try:
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(output_dir)
            print("✅ Extraction complete.")
        except Exception as e:
            print(f"❌ Failed to extract tar file: {e}")

    # 3. Single Gzip files (.gz but not a tarball)
    elif filename.endswith(".gz"):
        print(f"📦 Decompressing gzip file: {os.path.basename(file_path)}...")
        try:
            out_file_path = file_path[:-3]  # Remove .gz extension
            with gzip.open(file_path, "rb") as f_in:
                with open(out_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✅ Decompressed to {os.path.basename(out_file_path)}.")
        except Exception as e:
            print(f"❌ Failed to decompress gzip file: {e}")


def download_and_extract_zenodo(record_url, output_dir="./zenodo_download"):
    """Fetches Zenodo record metadata, streams file downloads, and decompresses archives."""
    record_id = extract_record_id(record_url)
    api_url = f"https://zenodo.org/api/records/{record_id}"

    print(f"🔍 Fetching metadata for Zenodo record {record_id}...")
    response = requests.get(api_url)

    # Fallback to the dedicated files endpoint if the primary record layout is missing files
    if response.status_code != 200:
        api_url = f"https://zenodo.org/api/records/{record_id}/files"
        response = requests.get(api_url)
        if response.status_code != 200:
            print(
                f"❌ Error: Could not retrieve record metadata (Status code: {response.status_code})"
            )
            return

    data = response.json()
    file_entries = []

    # Parse file metadata safely across different API formats (InvenioRDM vs older structures)
    if "files" in data:
        files_obj = data["files"]
        if isinstance(files_obj, list):
            file_entries = files_obj
        elif isinstance(files_obj, dict):
            entries = files_obj.get("entries", {})
            file_entries = (
                list(entries.values()) if isinstance(entries, dict) else entries
            )
    elif "entries" in data:
        file_entries = (
            list(data["entries"].values())
            if isinstance(data["entries"], dict)
            else data["entries"]
        )

    if not file_entries:
        print("⚠️ No files found in this record (it might be a metadata-only record).")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"🚀 Found {len(file_entries)} file(s). Starting downloads to '{output_dir}'...\n"
    )

    for entry in file_entries:
        # Resolve filename and download links gracefully across schemas
        filename = entry.get("key") or entry.get("filename") or entry.get("id")

        download_url = None
        if "links" in entry:
            download_url = entry["links"].get("content") or entry["links"].get("self")
        if not download_url:
            download_url = entry.get("download") or entry.get("links", {}).get(
                "download"
            )

        if not filename or not download_url:
            print(f"⚠️ Skipping unrecognizable file entry: {entry}")
            continue

        file_path = os.path.join(output_dir, filename)
        print(f"📥 Downloading: {filename}...")

        # Stream the download chunk-by-chunk to save RAM on large datasets
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"💾 Downloaded successfully.")

            # Check and run decompression
            extract_archive(file_path, output_dir)
            print("-" * 50)

        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            print("-" * 50)


if __name__ == "__main__":
    download_and_extract_zenodo("https://zenodo.org/record/20450254", "./evaluation")
