import argparse          # Standard library module for parsing command-line arguments
import json              # Used to serialize the verbose_json response to a file
import os                # Used to read environment variables
import re                # Regular expressions, used to parse Craig's track file names
import shutil            # shutil.which() checks that ffmpeg is installed
import subprocess        # Runs ffmpeg to convert .aac tracks
import tempfile          # Temporary folder for converted tracks / extracted zips
import zipfile           # Craig downloads come as .zip archives
from pathlib import Path # Object-oriented file paths (easy to swap file extensions)
from typing import Callable

from dotenv import load_dotenv  # Reads KEY=VALUE pairs from a .env file into os.environ
from openai import OpenAI       # Groq exposes an OpenAI-compatible API, so the OpenAI SDK works

# rich draws the colored progress bars, panels and tables in the terminal
from rich.console import Console
from rich.markup import escape  # Makes user names like "[abc]" safe to print with rich markup
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
                           TextColumn, TimeElapsedColumn)
from rich.table import Column, Table

# Loads .env if it exists; does not crash if missing (useful in production,
# where variables are usually set directly in the environment instead)
load_dotenv()

# One shared console for all output, so prints and progress bars don't clash
console = Console()

# Optional file holding the transcription prompt (see build_params() for details)
PROMPT_FILE = "PROMPT_FILE"

# Folder where recordings live. Without an explicit input, the newest entry here is used.
RECORDINGS_DIR = "recordings"

# File types Groq accepts directly (taken from the API's own error message).
# Anything else (e.g. Craig's raw .aac tracks) is converted with ffmpeg first.
GROQ_FORMATS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".opus", ".wav", ".webm"}

# Extensions treated as audio when looking inside a multi-track folder.
AUDIO_EXTENSIONS = GROQ_FORMATS | {".aac"}

# Groq's free-tier upload limit (dev tier allows 100 MB).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Settings for cutting silence out of multi-track recordings
SAMPLE_RATE = 16000     # Groq downsamples to 16 kHz mono anyway, so we do it up front
BYTES_PER_SAMPLE = 2    # 16-bit PCM
MIN_SILENCE = 1.0       # Seconds of quiet needed to count as a pause (shorter = within speech)
PADDING = 0.3           # Seconds kept before/after each speech region so words aren't clipped
GAP = 1.0               # Seconds of silence inserted between kept regions, so Whisper sees pauses
MAX_CHUNK_SECONDS = 20 * 60  # Longest condensed audio per upload: ~14 MB of FLAC, under the 25 MB limit

# A progress callback receives "seconds of audio processed so far"
ProgressCallback = Callable[[float], None] | None


def get_env_var(name: str) -> str:
    """Return the value of a required environment variable or fail loudly."""
    value = os.getenv(name)  # Returns None if the variable isn't set
    if not value:  # Catches both None and empty strings ("")
        raise RuntimeError(f"Missing required environment variable: {name}. "
                           f"Copy .env.example to .env and fill it in.")
    return value


def find_latest_recording(directory: str = RECORDINGS_DIR) -> Path:
    """Return the most recently modified recording in `directory`.

    A recording is either a single audio file, a .zip from Craig, or a folder
    (e.g. an extracted Craig multi-track download).
    """
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"No {directory}/ folder found. Create it and put recordings there, "
                                f"or pass a file/folder path as an argument.")
    candidates = [
        p for p in Path(directory).iterdir()
        if p.is_dir() or p.suffix.lower() in AUDIO_EXTENSIONS | {".zip"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No recordings found in {Path(directory).resolve()}")
    # os.path.getmtime returns the last-modification timestamp (seconds since epoch);
    # max() with key= picks the entry with the largest, i.e. newest, timestamp.
    return max(candidates, key=os.path.getmtime)


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line options (the model "hyperparameters")."""
    parser = argparse.ArgumentParser(description="Transcribe audio with Groq Whisper.")

    # Positional argument: an audio file, a Craig .zip, or a multi-track folder.
    # nargs="?" makes it optional; when omitted, main() picks the newest entry in recordings/.
    parser.add_argument("audio", nargs="?", default=None,
                        help=f"Audio file, Craig .zip or multi-track folder "
                             f"(default: most recently modified entry in {RECORDINGS_DIR}/)")

    # Which Whisper model to use. "turbo" is faster/cheaper, the full model is more accurate.
    parser.add_argument("--model", default="whisper-large-v3",
                        help="Model name (e.g. whisper-large-v3, whisper-large-v3-turbo)")

    # Telling the model the spoken language up front improves accuracy and speed.
    # Defaults to Polish. Pass an empty string (--language "") to let the
    # model auto-detect the language instead.
    parser.add_argument("--language", default="pl",
                        help="ISO-639-1 language code, e.g. 'en', 'pl' (default: pl; \"\" = auto-detect)")

    # Randomness of decoding. 0 = deterministic (most likely text);
    # higher values allow more variation. type=float converts the string input.
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0-1)")

    # Shape of the API response (single-file mode only; multi-track always
    # uses verbose_json because merging needs the timestamps):
    #   json         -> object with just the transcript text
    #   verbose_json -> text plus segments, timestamps, detected language, etc.
    #   text         -> plain string
    # `choices` makes argparse reject any other value with a helpful error.
    parser.add_argument("--response-format", default="verbose_json",
                        choices=["json", "verbose_json", "text"], help="API response format")

    # Multi-track only. Each speaker's track is mostly silence, and Whisper
    # "hallucinates" text in silence (on these recordings: endless "Dziękuję.").
    # Groq returns no_speech_prob = 0 even for pure silence, so the output can't
    # be filtered afterwards; instead silence is cut out *before* uploading.
    # Audio quieter than this level (in dB) counts as silence. Discord tracks
    # are near-digital silence between utterances, so -50 dB works well; raise
    # it (e.g. -40) if background noise is being treated as speech.
    parser.add_argument("--silence-db", type=float, default=-50.0,
                        help="Multi-track: loudness below which audio counts as silence (default: -50)")

    # argparse turns "--response-format" into the attribute `response_format`
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> dict:
    """Collect the API parameters shared by every transcription request."""
    # Always-present options go in directly...
    params = {
        "model": args.model,
        "temperature": args.temperature,
        "response_format": args.response_format,
    }
    # ...while optional ones are only added when the user supplied them,
    # so the API uses its own defaults otherwise (e.g. language auto-detection).
    if args.language:
        params["language"] = args.language

    # The prompt is a text hint the model conditions on, e.g. names, jargon, or a
    # desired punctuation style. It helps with spelling of uncommon words.
    # It is read from PROMPT_FILE in the current directory, if that file exists.
    prompt_path = Path(PROMPT_FILE)
    if prompt_path.is_file():
        prompt = prompt_path.read_text(encoding="utf-8").strip()  # strip() drops trailing newlines
        if prompt:  # Skip an empty file instead of sending an empty prompt
            params["prompt"] = prompt
    return params


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def run_ffmpeg(args: list[str], on_progress: ProgressCallback = None) -> str:
    """Run ffmpeg with `args`, reporting progress; return ffmpeg's log text.

    "-progress pipe:1" makes ffmpeg write machine-readable status lines such as
    "out_time_us=12345678" (microseconds of audio processed) to stdout, which we
    read live to move the progress bar. The normal log (stderr) goes to a temp
    file so it can't fill up a pipe and stall ffmpeg; it's returned afterwards
    because silencedetect writes its results there.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for this recording but isn't installed "
                           "(e.g. `sudo apt install ffmpeg` or `brew install ffmpeg`)")

    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y", "-progress", "pipe:1", *args]
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
        # stdout=PIPE guarantees proc.stdout is set; the assert tells the type checker
        assert proc.stdout is not None
        for line in proc.stdout:  # Yields lines as ffmpeg writes them
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit() and on_progress:
                on_progress(int(value) / 1_000_000)
        proc.wait()
        log.seek(0)
        text = log.read().decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{text[-2000:]}")
    return text


def adts_duration(path: Path) -> float:
    """Exact duration of a raw .aac (ADTS) file, by counting its audio frames.

    Raw AAC has no duration header, and ffprobe only *estimates* it from the
    bitrate, which on Craig's tracks is off by up to 3x. Every ADTS frame
    starts with a small header that says how long the frame is (in bytes) and
    its sample rate, and holds 1024 samples per data block, so walking the
    headers gives the exact length. It's fast: no audio is decoded.
    """
    sample_rates = [96000, 88200, 64000, 48000, 44100, 32000, 24000,
                    22050, 16000, 12000, 11025, 8000, 7350]
    data = path.read_bytes()
    i, seconds = 0, 0.0
    while i + 7 <= len(data):
        if data[i] != 0xFF or (data[i + 1] & 0xF0) != 0xF0:
            break  # Not a frame header (corrupt or trailing data): stop counting
        rate_index = (data[i + 2] >> 2) & 0x0F
        frame_length = ((data[i + 3] & 0x03) << 11) | (data[i + 4] << 3) | (data[i + 5] >> 5)
        # Guard against corrupt headers: a length below the 7-byte header size
        # would never advance i (endless loop), and indexes 13-15 aren't valid rates
        if frame_length < 7 or rate_index >= len(sample_rates):
            break
        rate = sample_rates[rate_index]
        blocks = (data[i + 6] & 0x03) + 1
        seconds += blocks * 1024 / rate
        i += frame_length
    return seconds


def audio_duration(path: Path) -> float | None:
    """Duration in seconds (None if unknown), used as the progress bar total."""
    if path.suffix.lower() == ".aac":
        return adts_duration(path)
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None  # The progress bar will just pulse instead of filling up


def to_groq_format(audio_path: Path, work_dir: Path, on_progress: ProgressCallback = None) -> Path:
    """Return a version of `audio_path` that Groq accepts, converting if needed."""
    if audio_path.suffix.lower() in GROQ_FORMATS:
        return audio_path  # Already fine, upload as-is

    out_path = work_dir / (audio_path.stem + ".m4a")
    # Remux instead of re-encoding: "-c:a copy" puts the same AAC audio into an
    # .m4a (MP4) container, so it's fast and loses no quality. The aac_adtstoasc
    # filter adapts the raw ADTS stream headers to what MP4 expects.
    run_ffmpeg(["-i", str(audio_path), "-c:a", "copy", "-bsf:a", "aac_adtstoasc", str(out_path)],
               on_progress)
    return out_path


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(client: OpenAI, audio_path: Path, params: dict):
    """Upload one file to Groq and return the API response."""
    size = audio_path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        console.print(f"[yellow]⚠ {escape(audio_path.name)} is {size / 2**20:.1f} MB, "
                      f"above the free-tier 25 MB limit[/]")

    # Open the audio in binary mode ("rb") and upload it. `**params` unpacks
    # the dict into keyword arguments: model=..., temperature=..., etc.
    # The `with` block closes the file automatically afterwards.
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(file=f, **params)


def speaker_from_filename(path: Path) -> str:
    """Craig names tracks "<number>-<username>.aac"; return just the username."""
    # ^\d+- matches the leading track number and dash, e.g. "3-" in "3-alice"
    return re.sub(r"^\d+-", "", path.stem)


def format_timestamp(seconds: float) -> str:
    """Turn 754.2 into "00:12:34"."""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def detect_speech(track: Path, silence_db: float, duration: float | None,
                  on_progress: ProgressCallback = None) -> list[tuple[float, float]]:
    """Return (start, end) times in seconds of the parts of `track` with speech.

    `duration` is the track length (None if unknown); it's used to recognise a
    silence that runs right to the end of the file.
    """
    # ffmpeg's silencedetect filter logs "silence_start: X" / "silence_end: Y".
    # "-f null -" decodes the whole file without writing any output.
    log = run_ffmpeg(["-i", str(track), "-af", f"silencedetect=noise={silence_db}dB:d={MIN_SILENCE}",
                      "-f", "null", "-"], on_progress)
    # ffmpeg can log a slightly negative start (e.g. -0.0213) for silence at
    # the very beginning, so the pattern allows a minus sign and the value is
    # clamped to 0. Missing it would shift every start/end pair out of step.
    number = r"(-?\d+(?:\.\d+)?)"
    starts = [max(0.0, float(x)) for x in re.findall(rf"silence_start: {number}", log)]
    ends = [max(0.0, float(x)) for x in re.findall(rf"silence_end: {number}", log)]

    # Speech is everything *between* silences. Build the list of silences
    # (a silence running to the end of the file has no silence_end) and invert it.
    silences = list(zip(starts, ends + [float("inf")] * (len(starts) - len(ends))))
    speech, cursor = [], 0.0
    for s_start, s_end in silences:
        if s_start > cursor:
            speech.append((cursor, s_start))
        cursor = s_end
    # After the last silence there's speech until the end of the file, unless
    # that silence itself reached the end. (ffmpeg 6+ logs a silence_end at the
    # end of the file, so cursor is then the track length rather than inf.)
    # Without this check a track that is silent at the end, or entirely silent,
    # would get a junk region of pure silence and never be skipped.
    reaches_end = cursor == float("inf") or (duration is not None and cursor >= duration - 0.05)
    if not reaches_end:
        speech.append((cursor, float("inf")))

    # Pad each region a little and merge regions that now overlap
    merged: list[tuple[float, float]] = []
    for start, end in speech:
        start, end = max(0.0, start - PADDING), end + PADDING
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def decode_pcm(track: Path, work_dir: Path, on_progress: ProgressCallback = None) -> bytes:
    """Decode `track` to raw 16 kHz mono 16-bit PCM ("-f s16le") and return the bytes.

    This takes about 1.9 MB per minute: ~48 MB for 25 minutes, ~350 MB for 3 hours.
    """
    pcm_path = work_dir / f"{track.stem}.pcm"
    run_ffmpeg(["-i", str(track), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", str(pcm_path)],
               on_progress)
    pcm = pcm_path.read_bytes()
    pcm_path.unlink()  # Free the disk space; the data is in memory now
    return pcm


def plan_chunks(regions: list[tuple[float, float]], total_seconds: float) -> list[list[tuple[float, float]]]:
    """Group speech regions into chunks whose condensed audio fits one upload.

    Groq's free tier accepts 25 MB per file. 16 kHz mono FLAC is about 0.7 MB
    per minute of speech, so a talkative person in a long call can go over that.
    Each chunk holds at most MAX_CHUNK_SECONDS of audio and is cut between
    speech regions, so no words are split. (A single region longer than that,
    i.e. a monologue with no 1-second pause, is cut into pieces.)
    """
    chunks: list[list[tuple[float, float]]] = [[]]
    length = 0.0  # Condensed length of the chunk being filled
    for start, end in regions:
        end = min(end, total_seconds)  # "inf" means "until the end of the file"
        while end - start > 0:
            piece_end = min(end, start + MAX_CHUNK_SECONDS)
            piece_length = piece_end - start + GAP
            if chunks[-1] and length + piece_length > MAX_CHUNK_SECONDS:
                chunks.append([])  # This piece doesn't fit, start a new chunk
                length = 0.0
            chunks[-1].append((start, piece_end))
            length += piece_length
            start = piece_end
    return [c for c in chunks if c]


def write_condensed(pcm: bytes, regions: list[tuple[float, float]],
                    out_path: Path) -> tuple[list[tuple[float, float]], float]:
    """Write only the speech `regions` of `pcm` to `out_path` as 16 kHz mono FLAC.

    Returns (time_map, duration). time_map holds, for each kept region,
    (position in condensed file, position in original file) and is used later
    to translate timestamps back. duration is the condensed length in seconds.
    """
    def to_byte(seconds: float) -> int:
        # Convert a time to a byte offset, clamped to the audio length
        return min(len(pcm), int(seconds * SAMPLE_RATE) * BYTES_PER_SAMPLE)

    gap = b"\x00" * to_byte(GAP)  # Digital silence (all-zero samples)
    view = memoryview(pcm)  # Slicing a memoryview doesn't copy the audio
    time_map, condensed_pos = [], 0.0
    pcm_path = out_path.with_suffix(".pcm")

    # Write each piece straight to disk instead of joining them in memory,
    # so a long track isn't held in memory twice
    with open(pcm_path, "wb") as f:
        for start, end in regions:
            chunk = view[to_byte(start):to_byte(end)]
            time_map.append((condensed_pos, start))
            f.write(chunk)
            f.write(gap)
            condensed_pos += (len(chunk) + len(gap)) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    # Encode to FLAC: lossless and roughly a third of the size of raw PCM.
    # (Opus would be ~5x smaller still, but in a test it dropped ~8% of the
    # words and cut off Polish word endings, so lossless it is.)
    # "-f s16le -ar -ac" before "-i" tells ffmpeg how to read the raw samples.
    run_ffmpeg(["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", str(pcm_path), str(out_path)])
    pcm_path.unlink()  # The raw file is no longer needed
    return time_map, condensed_pos


def region_index(t: float, time_map: list[tuple[float, float]]) -> int:
    """Return which kept region a condensed-file timestamp `t` falls into."""
    # The last region that starts at or before t. (A time inside an inserted
    # gap counts as belonging to the region before it.)
    index = 0
    for i, (condensed_start, _) in enumerate(time_map):
        if condensed_start > t:
            break
        index = i
    return index


def to_original_time(t: float, time_map: list[tuple[float, float]], index: int) -> float:
    """Translate condensed-file time `t` in region `index` back to the original recording."""
    condensed_start, original_start = time_map[index]
    return original_start + (t - condensed_start)


def words_to_utterances(words: list[dict], time_map: list[tuple[float, float]]) -> list[dict]:
    """Group word timestamps into one utterance per speech region.

    Whisper's *segments* can span several regions (it happily joins speech from
    minute 2 and minute 23 once the silence between is cut out), which would
    give later words a wrong timestamp. Words each have their own timestamp, so
    assigning words to regions keeps every utterance at its true time.
    """
    utterances = []
    current_index = None
    for w in words:
        index = region_index(w["start"], time_map)
        if index != current_index:  # A new region starts a new utterance
            utterances.append({"start": to_original_time(w["start"], time_map, index), "words": []})
            current_index = index
        utterances[-1]["words"].append(w["word"].strip())
        utterances[-1]["end"] = to_original_time(w["end"], time_map, index)
    return [
        {"start": round(u["start"], 2), "end": round(u["end"], 2), "text": " ".join(u["words"])}
        for u in utterances
    ]


def make_progress() -> Progress:
    """Build the progress display: spinner, name, bar, percent, status, elapsed time."""
    return Progress(
        SpinnerColumn("dots", style="magenta"),
        TextColumn("{task.description}", table_column=Column(width=22, no_wrap=True)),
        BarColumn(bar_width=32, style="grey23", complete_style="magenta",
                  finished_style="green", pulse_style="cyan"),
        TaskProgressColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )


def transcribe_multitrack(client: OpenAI, tracks: list[Path], params: dict,
                          work_dir: Path, silence_db: float) -> tuple[list[dict], list[dict]]:
    """Transcribe each speaker's track and merge the segments by start time.

    Craig's tracks all start at the beginning of the recording (silence is
    included while someone isn't talking), so after mapping timestamps back to
    the original timeline they line up and can be merged directly.

    Returns (segments, stats): the merged conversation and per-speaker numbers
    for the summary table.
    """
    # Timestamps only exist in verbose_json, so force it here, and ask for a
    # timestamp per word (see words_to_utterances for why)
    params = {**params, "response_format": "verbose_json",
              "timestamp_granularities": ["word", "segment"]}

    segments, stats = [], []
    steps_per_track = 3  # detect speech -> cut silence -> transcribe

    with make_progress() as progress:
        # The overall bar counts finished steps across all tracks
        overall = progress.add_task("[bold]Overall[/]", total=len(tracks) * steps_per_track,
                                    status="")
        # One row per track, added up front so you can see the whole queue
        rows = {
            track: progress.add_task(f"  [cyan]{escape(speaker_from_filename(track))}[/]",
                                     total=None, start=False, status="[dim]waiting[/]")
            for track in tracks
        }

        for track in tracks:
            speaker = speaker_from_filename(track)
            row = rows[track]
            duration = audio_duration(track)

            def advance_to(seconds: float, row=row) -> None:
                # Called by ffmpeg as it works through the audio
                progress.update(row, completed=seconds)

            # Step 1: find where this person speaks
            progress.start_task(row)
            progress.update(row, total=duration, completed=0, status="[yellow]detecting speech[/]")
            regions = detect_speech(track, silence_db, duration, advance_to)
            progress.advance(overall)

            if not regions:
                progress.update(row, total=1, completed=1, status="[dim]no speech, skipped[/]")
                progress.advance(overall, steps_per_track - 1)
                continue  # This person never spoke, nothing to send

            # Step 2: cut the silence out, split into upload-sized chunks
            progress.update(row, total=duration, completed=0, status="[yellow]cutting silence[/]")
            pcm = decode_pcm(track, work_dir, advance_to)
            total_seconds = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            chunks = []  # (condensed file, time_map, condensed seconds) per chunk
            for n, chunk_regions in enumerate(plan_chunks(regions, total_seconds), start=1):
                out = work_dir / f"{track.stem}_{n}.flac"
                chunks.append((out, *write_condensed(pcm, chunk_regions, out)))
            del pcm  # Release the decoded track before uploading
            kept_seconds = sum(seconds for _, _, seconds in chunks)
            progress.advance(overall)

            # Step 3: upload and wait for Groq. There's no way to know how far
            # along the server is, so total=None makes the bar pulse instead.
            utterances = []
            for n, (condensed, time_map, seconds) in enumerate(chunks, start=1):
                part = f" (part {n}/{len(chunks)})" if len(chunks) > 1 else ""
                progress.update(row, total=None,
                                status=f"[cyan]transcribing {seconds / 60:.1f} min on Groq{part} ☁[/]")
                result = transcribe(client, condensed, params).model_dump()
                utterances += [u for u in words_to_utterances(result.get("words") or [], time_map)
                               if u["text"]]
            progress.advance(overall)

            segments += [{"speaker": speaker, **u} for u in utterances]
            stats.append({
                "speaker": speaker,
                "speech": kept_seconds,
                "utterances": len(utterances),
                "words": sum(len(u["text"].split()) for u in utterances),
            })
            progress.update(row, total=1, completed=1,
                            status=f"[green]✓ {len(utterances)} utterances[/]")

        progress.update(overall, status="[bold green]done ✨[/]")

    # Sort everyone's segments into one chronological conversation
    segments.sort(key=lambda s: s["start"])
    return segments, stats


def format_conversation(segments: list[dict]) -> str:
    """Render merged segments as "[hh:mm:ss] speaker: text" lines.

    Consecutive segments from the same speaker are joined into one line, so a
    long statement isn't split across many lines.
    """
    lines = []
    current = None  # The line being built: {"speaker", "start", "texts"}
    for seg in segments:
        if current and current["speaker"] == seg["speaker"]:
            current["texts"].append(seg["text"])
        else:
            if current:
                lines.append(current)
            current = {"speaker": seg["speaker"], "start": seg["start"], "texts": [seg["text"]]}
    if current:
        lines.append(current)

    return "\n".join(
        f"[{format_timestamp(line['start'])}] {line['speaker']}: {' '.join(line['texts'])}"
        for line in lines
    ) + "\n"


def print_summary(stats: list[dict]) -> None:
    """Show a table of who spoke how much."""
    table = Table(title="Speakers", title_style="bold magenta", border_style="grey50")
    table.add_column("Speaker", style="cyan")
    table.add_column("Speech", justify="right")
    table.add_column("Share", justify="left")
    table.add_column("Utterances", justify="right")
    table.add_column("Words", justify="right")

    total = sum(s["speech"] for s in stats) or 1  # "or 1" avoids dividing by zero
    for s in sorted(stats, key=lambda s: s["speech"], reverse=True):
        share = s["speech"] / total
        bar = "█" * round(share * 20)  # A tiny inline bar chart of speaking time
        table.add_row(escape(s["speaker"]), format_timestamp(s["speech"]),
                      f"[magenta]{bar}[/] {share:.0%}", str(s["utterances"]), str(s["words"]))
    console.print(table)


def main() -> None:
    args = parse_args()
    if args.audio and not Path(args.audio).exists():
        raise FileNotFoundError(f"Input not found: {args.audio}")

    # Point the OpenAI client at Groq's endpoint and authenticate with the Groq key.
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=get_env_var("GROQ_API_KEY"))
    params = build_params(args)

    # Use the path given on the command line, or fall back to the newest recording
    input_path = Path(args.audio) if args.audio else find_latest_recording()

    # A header box summarizing what's about to happen
    console.print(Panel.fit(
        f"[bold]{escape(input_path.name)}[/]\n"
        f"model [cyan]{params['model']}[/] · language [cyan]{params.get('language', 'auto')}[/] · "
        f"prompt {'[green]yes[/]' if 'prompt' in params else '[dim]no[/]'}",
        title="🎙  Groq transcription", border_style="magenta",
    ))

    # Outputs are named after the input: recording.m4a / recording.zip /
    # folder "recording" -> recording.txt (and recording.json).
    # with_name keeps the folder and swaps the name, so it works for folders too.
    txt_path = input_path.with_name(input_path.stem + ".txt")
    json_path = input_path.with_name(input_path.stem + ".json")

    # A temporary folder for converted tracks and extracted zips; it's deleted
    # automatically when the `with` block ends.
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # A Craig .zip is extracted first and then handled like a folder
        source = input_path
        if input_path.suffix.lower() == ".zip":
            source = work_dir / "extracted"
            with console.status("Extracting zip…"):
                with zipfile.ZipFile(input_path) as zf:
                    zf.extractall(source)

        if source.is_dir():
            # Multi-track: one audio file per speaker. sorted() keeps Craig's
            # numbering order (1-..., 2-...). info.txt and raw.dat are skipped
            # because they don't have audio extensions.
            tracks = sorted(p for p in source.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
            if not tracks:
                raise FileNotFoundError(f"No audio tracks found in {input_path}")

            segments, stats = transcribe_multitrack(client, tracks, params, work_dir, args.silence_db)
            txt_path.write_text(format_conversation(segments), encoding="utf-8")
            # ensure_ascii=False keeps non-English characters (e.g. Polish ą, ę) readable
            json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

            print_summary(stats)
            console.print(f"[green]✓[/] Transcript: [bold]{escape(str(txt_path))}[/]")
            console.print(f"[green]✓[/] Details:    [bold]{escape(str(json_path))}[/]")
            return

        # Single file (e.g. Craig's mixed single-track download)
        with make_progress() as progress:
            row = progress.add_task(f"[cyan]{escape(source.name)}[/]", total=None, status="")
            if source.suffix.lower() not in GROQ_FORMATS:
                progress.update(row, total=audio_duration(source), status="[yellow]converting[/]")
                source = to_groq_format(source, work_dir, lambda s: progress.update(row, completed=s))
            progress.update(row, total=None, status="[cyan]transcribing on Groq ☁[/]")
            result = transcribe(client, source, params)
            progress.update(row, total=1, completed=1, status="[green]✓ done[/]")

    if args.response_format == "verbose_json":
        # Keep all the extra details (segments, timestamps) as a JSON file.
        out_path = json_path
        # model_dump() converts the SDK response object into a plain dict
        content = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        # For json/text we only need the transcript
        out_path = txt_path
        # With "text" format the SDK returns a plain str; with "json" it returns
        # an object whose transcript lives in the .text attribute.
        content = result if isinstance(result, str) else result.text

    # Write as UTF-8 so non-ASCII characters are saved correctly on any OS.
    out_path.write_text(content, encoding="utf-8")
    console.print(f"[green]✓[/] Transcription written to [bold]{escape(str(out_path))}[/]")


# Run main() only when the file is executed directly (python main.py),
# not when it is imported as a module from another script.
if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as e:
        # Expected problems (missing key, file, ffmpeg...) get a short red
        # message instead of a long Python traceback
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/]")
        raise SystemExit(130)  # Conventional exit code for Ctrl+C
