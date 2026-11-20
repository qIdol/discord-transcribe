(vibe-coded)
# discord-transcribe

Transcribe Discord calls recorded with the [Craig](https://craig.chat) bot, or any audio file, using [Groq's Whisper API](https://console.groq.com/docs/speech-to-text).

With a Craig **multi-track** recording (one audio file per person), you get a transcript with speaker names and timestamps:

```text
[00:01:04] alice: Hej, słyszycie mnie?
[00:01:05] bob: Tak, słychać. Dobra, to chyba wszyscy są.
[00:01:08] alice: Super.
[00:01:11] bob: To zaczynajmy. Czy ktoś pracuje na Macu?
```

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs the right Python version (3.14) and the dependencies for you.
- A Groq API key. Create one for free at <https://console.groq.com/keys>.
- [ffmpeg](https://ffmpeg.org/download.html), needed for multi-track recordings and `.aac` files:
  - Ubuntu/Debian/WSL: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install ffmpeg`

## Setup

```bash
git clone https://github.com/qIdol/discord-transcribe.git
cd groq-transcriptions
cp .env.example .env              # then put your Groq API key in .env
cp PROMPT_FILE.example PROMPT_FILE   # optional, see "Prompt" below
```

You don't need a separate install step: `uv run` creates the environment the first time you run it.

## Usage

### Craig multi-track recording (recommended)

1. When the Craig recording finishes, open its download page and choose **Multi-track → AAC**.
2. Put the downloaded `.zip` in `recordings/`. You can also unzip it there; a folder works too.
3. Run:

   ```bash
   uv run main.py
   ```

To choose a recording yourself instead of the newest one, pass its path:

```bash
uv run main.py recordings/craig-abc123.zip
uv run main.py recordings/my_meeting/
```

### Single audio file

Craig's mixed single-track download or any other recording works too. You just won't get speaker names.

```bash
uv run main.py recordings/meeting.m4a
```

### Output

Output files go next to the input and use its name:

| Input | Output |
| --- | --- |
| Multi-track folder or `.zip` | `name.txt` (readable conversation) and `name.json` (every utterance with speaker, start and end time) |
| Single file, `--response-format verbose_json` | `name.json`: full Groq response with segments and timestamps |
| Single file, `--response-format json` or `text` | `name.txt`: plain transcript |

Existing output files with the same name are overwritten.

## Defaults

Running `uv run main.py` with no options:

- **Input:** the most recently modified file or folder in `recordings/`
- **Model:** `whisper-large-v3`
- **Language:** Polish (`pl`)
- **Temperature:** `0.0`
- **Response format (single file):** `verbose_json`, saved as `name.json`
- **Silence threshold (multi-track):** `-50` dB
- **Prompt:** the contents of `PROMPT_FILE`, if that file exists

Each of these can be changed with the options below.

## Options

```bash
uv run main.py --help
```

| Option | Description |
| --- | --- |
| `audio` (positional) | Audio file, Craig `.zip`, or multi-track folder |
| `--model` | `whisper-large-v3` is the most accurate; `whisper-large-v3-turbo` is faster and cheaper |
| `--language` | Spoken language as an [ISO-639-1 code](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) (`en`, `de`, ...). Use `--language ""` to auto-detect |
| `--temperature` | Randomness, 0–1. `0` gives the most likely transcription |
| `--response-format` | Single-file mode only: `json`, `verbose_json` or `text` |
| `--silence-db` | Multi-track only: audio quieter than this counts as silence. Try `-40` if background noise is transcribed as speech |

## Prompt

If a file named `PROMPT_FILE` exists in the folder you run the script from, its contents are sent to Whisper as a prompt. Use it to improve the spelling of names, jargon and product names. Copy `PROMPT_FILE.example` to start.

Tips:

- Write the prompt **in the same language as the audio**. Otherwise Whisper may switch languages or translate.
- Write it like a sentence from the transcript, not an instruction. "Rozmowa o Grid2Op i Pythonie." works better than "Transcribe accurately."
- List the names and terms you expect to hear.
- Keep it short. Whisper only uses about the last 224 tokens.

`PROMPT_FILE` is in `.gitignore`, because prompts usually contain meeting-specific names.

## How multi-track transcription works

Each speaker's Craig track covers the whole call, so most of it is silence. Sending tracks as they are causes two problems:

- Whisper invents text in long silences (in Polish, endless "Dziękuję.").
- Groq gives no reliable way to filter that text out afterwards.

So, for each track the script:

1. **Finds where the person speaks**, using ffmpeg's `silencedetect`.
2. **Cuts out the silence** and keeps only the speech, with a 1-second pause between pieces. It saves the result as 16 kHz mono FLAC, which is the format Groq recommends. Uploads shrink a lot: a 25-minute track where someone talks for 20 seconds becomes a 20-second upload.
3. **Transcribes it with per-word timestamps**, then maps each word back to its time in the original recording.
4. **Merges all speakers** into one chronological conversation.

Craig starts every track at the beginning of the recording, so all the timestamps share the same timeline.

## Limitations

- **Upload size:** Groq's free tier accepts files up to 25 MB (100 MB on the dev tier). Multi-track uploads are small because the silence is removed. A long single-file recording can go over the limit, in which case the script prints a warning.
- **Timing of very short replies:** they're sometimes split oddly, because Whisper's word timing is imprecise at the points where silence was cut out.
- **People talking over each other:** in multi-track mode each person has their own track, so this works fine. In single-file mode it can reduce accuracy.

## Privacy

Audio is uploaded to Groq for transcription. Make sure everyone on the call agrees to being recorded and transcribed. `recordings/`, `.env` and `PROMPT_FILE` are git-ignored, so recordings, transcripts and your API key aren't committed by accident.

## License

[MIT](LICENSE)
