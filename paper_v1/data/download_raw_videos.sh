#!/usr/bin/env bash
# =============================================================================
# Download + prepare the 12 v19 source videos for the multi-resolution VMAF ladder.
# Produces (1080p/720p, audio stripped, H.264) references under raw_videos/.
#
# Roster matches configs/videos.py (12 titles, 9 train + 3 held-out).
#
# Usage:
#   cd new
#   bash data/download_raw_videos.sh
#   # then encode / VMAF / SI-TI pipeline
#
# Env vars: RAW_DIR (default data/raw_videos), TMP (default .raw_downloads),
#           PYTHON (default python), CRF (default 12), KEEP_TMP=1 to keep downloads.
# Requires: ffmpeg, wget or curl, python (zipfile), xz (for elephants_dream).
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RAW_DIR="${RAW_DIR:-data/raw_videos}"
TMP="${TMP:-.raw_downloads}"
PY="${PYTHON:-python}"
CRF="${CRF:-12}"
DERF_Y4M="https://media.xiph.org/video/derf/y4m"
mkdir -p "${RAW_DIR}" "${TMP}"

command -v ffmpeg >/dev/null 2>&1 || { echo "[FATAL] ffmpeg not found."; exit 2; }

fetch() {  # fetch <url> <out>
    local url="$1" out="$2"
    if [ -f "${out}" ]; then echo "  cached: ${out}"; return 0; fi
    echo "  downloading: ${url}"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${out}.part" "${url}" && mv "${out}.part" "${out}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "${out}.part" "${url}" && mv "${out}.part" "${out}"
    else
        echo "[FATAL] neither wget nor curl is available."; exit 2
    fi
}

unzip_one() {  # unzip_one <zip> <destdir>
    "${PY}" -m zipfile -e "$1" "$2"
}

find_first() {  # find_first <dir> <ext>
    find "$1" -type f -iname "*.$2" 2>/dev/null | head -n 1
}

transcode_ref() {  # transcode_ref <in> <out> [ffmpeg input opts before -i]
    local out="$1"; shift
    ffmpeg -y -hide_banner -loglevel error "$@" \
        -c:v libx264 -crf "${CRF}" -preset medium -pix_fmt yuv420p -an "${out}"
}

y4m_ref() {  # y4m_ref <slug> <y4m_url> [extra ffmpeg args e.g. -t 192]
    local slug="$1" url="$2"; shift 2
    local out="${RAW_DIR}/${slug}.mp4"
    if [ -f "${out}" ]; then echo "[${slug}] exists, skip"; return 0; fi
    echo "[${slug}] y4m -> mp4"
    fetch "${url}" "${TMP}/${slug}.y4m"
    transcode_ref "${out}" -i "${TMP}/${slug}.y4m" "$@"
    echo "  -> ${out}"
}

blender_zip_mov() {  # blender_zip_mov <slug> <zip_url> <ss> <duration>
    local slug="$1" url="$2" ss="$3" dur="$4"
    local out="${RAW_DIR}/${slug}.mp4"
    if [ -f "${out}" ]; then echo "[${slug}] exists, skip"; return 0; fi
    echo "[${slug}] blender zip -> mp4 (ss=${ss}, t=${dur})"
    fetch "${url}" "${TMP}/${slug}.zip"
    unzip_one "${TMP}/${slug}.zip" "${TMP}/${slug}_unz"
    local src
    src="$(find_first "${TMP}/${slug}_unz" mov)"
    [ -n "${src}" ] || src="$(find_first "${TMP}/${slug}_unz" mp4)"
    [ -n "${src}" ] || { echo "[FATAL] no video inside ${slug} zip"; exit 1; }
    transcode_ref "${out}" -ss "${ss}" -i "${src}" -t "${dur}"
    echo "  -> ${out}"
}

echo "=============================================================="
echo "Preparing 12 v19 source videos -> ${RAW_DIR}/"
echo "=============================================================="

# --- Blender (6) -------------------------------------------------------------

# bigbuckbunny: 192 s clip (48 chunks @ 4 s) from t=60 s
blender_zip_mov "bigbuckbunny" \
    "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip" \
    60 192

# tearsofsteel: 120 s clip from t=120 s (replaces tearsofsteel_short)
blender_zip_mov "tearsofsteel" \
    "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip" \
    120 120

# sintel: official 1080p trailer (~52 s)
if [ ! -f "${RAW_DIR}/sintel.mp4" ]; then
    echo "[sintel] trailer mp4"
    fetch "https://download.blender.org/apricot/trailer/sintel_trailer-1080p.mp4" "${TMP}/sintel_src.mp4"
    transcode_ref "${RAW_DIR}/sintel.mp4" -i "${TMP}/sintel_src.mp4"
    echo "  -> ${RAW_DIR}/sintel.mp4"
else echo "[sintel] exists, skip"; fi

# elephants_dream: 120 s from compressed DERF y4m (7.1 GB .xz)
if [ ! -f "${RAW_DIR}/elephants_dream.mp4" ]; then
    echo "[elephants_dream] xz y4m -> mp4 (120 s)"
    command -v xz >/dev/null 2>&1 || { echo "[FATAL] xz required for elephants_dream"; exit 2; }
    fetch "${DERF_Y4M}/elephants_dream_1080p24.y4m.xz" "${TMP}/elephants_dream.y4m.xz"
    transcode_ref "${RAW_DIR}/elephants_dream.mp4" \
        -i pipe:0 -t 120 < <(xzcat "${TMP}/elephants_dream.y4m.xz")
    echo "  -> ${RAW_DIR}/elephants_dream.mp4"
else echo "[elephants_dream] exists, skip"; fi

# ducks: Xiph DERF 1080p50 (~10 s, 500 frames); same SVT "Ducks Take Off" clip
y4m_ref "ducks" "${DERF_Y4M}/ducks_take_off_1080p50.y4m"

# cosmos: Internet Archive 1080p (held-out Blender animation)
# Trim to 120 s — full film (~10 min) causes VMAF timeout on high rungs.
if [ ! -f "${RAW_DIR}/cosmos.mp4" ]; then
    echo "[cosmos] archive.org 1080p (trim 120s)"
    fetch "https://archive.org/download/CosmosLaundromatFirstCycle/Cosmos%20Laundromat%20-%20First%20Cycle%20%281080p%29.mp4" \
        "${TMP}/cosmos_src.mp4"
    transcode_ref "${RAW_DIR}/cosmos.mp4" -ss 0 -i "${TMP}/cosmos_src.mp4" -t 120
    echo "  -> ${RAW_DIR}/cosmos.mp4"
else echo "[cosmos] exists, skip"; fi

# --- Xiph DERF (6) -----------------------------------------------------------

y4m_ref "parkjoy"       "${DERF_Y4M}/park_joy_1080p50.y4m"
y4m_ref "old_town_cross" "${DERF_Y4M}/old_town_cross_1080p50.y4m"
y4m_ref "rush_hour"     "${DERF_Y4M}/rush_hour_1080p25.y4m"
y4m_ref "into_tree"     "${DERF_Y4M}/in_to_tree_1080p50.y4m"
y4m_ref "sunflower"     "${DERF_Y4M}/sunflower_1080p25.y4m"

# kristen_and_sara: 720p talking-head held-out (Meridian IMF impractical at ~96 GB)
y4m_ref "kristen_and_sara" "${DERF_Y4M}/KristenAndSara_1280x720_60.y4m"

echo ""
echo "=============================================================="
echo "Done. References in ${RAW_DIR}/:"
ls -lh "${RAW_DIR}"/*.mp4 2>/dev/null || true
echo ""
echo "Canonical list: python configs/videos.py"
if [ "${KEEP_TMP:-0}" != "1" ]; then
    echo "Removing temporary downloads in ${TMP} (set KEEP_TMP=1 to keep)."
    rm -rf "${TMP}"
fi
echo "Next: encode ladder -> VMAF -> SI/TI (see data/video_encoder.py)"
echo "=============================================================="
