# `af_musicdamper` integration guide (FFmpeg + sherpa-onnx C API)

This guide assumes:
- You have an FFmpeg source tree.
- You have built/installed `libsherpa-onnx-c-api` and `sherpa-onnx/c-api/c-api.h`.
- You want a CPU-only source-separation filter that outputs **vocals**.

## 1) Add the filter source file

Copy `c-api-examples/ffmpeg-af_musicdamper.c` from this repo into your FFmpeg tree:

```bash
cp /path/to/sherpa-onnx/c-api-examples/ffmpeg-af_musicdamper.c \
  /path/to/ffmpeg/libavfilter/af_musicdamper.c
```

## 2) Register it in `libavfilter/allfilters.c`

Add this extern declaration near other audio filter declarations:

```c
extern const AVFilter ff_af_musicdamper;
```

Then add it to the filter list in `allfilters.c`:

```c
&ff_af_musicdamper,
```

## 3) Hook it into `libavfilter/Makefile`

Append this object entry:

```make
OBJS-$(CONFIG_MUSICDAMPER_FILTER)            += af_musicdamper.o
```

## 4) Add configure switch (optional but recommended)

If you keep FFmpeg's per-filter configure style, add:

- In `configure`:

```bash
musicdamper_filter_deps="avfilter"
musicdamper_filter_extralibs="-lsherpa-onnx-c-api"
```

And include it in the list of enabled filters (pattern varies by FFmpeg version).

If you prefer quick local testing, you can inject extra libs directly:

```bash
./configure \
  --extra-cflags="-I/path/to/sherpa-onnx" \
  --extra-ldflags="-L/path/to/sherpa-onnx/build/lib" \
  --extra-libs="-lsherpa-onnx-c-api" \
  ...other FFmpeg flags...
```

> If the runtime loader cannot find the library, set:
>
> ```bash
> export LD_LIBRARY_PATH=/path/to/sherpa-onnx/build/lib:$LD_LIBRARY_PATH
> ```

## 5) Build FFmpeg

```bash
make -j$(nproc)
```

## 6) Filter options

`musicdamper` supports:

- `model_type` = `uvr` (default) or `spleeter`
- `uvr_model` = path to UVR MDX-Net ONNX (required for `model_type=uvr`)
- `spleeter_vocals` = path to vocals ONNX (required for `model_type=spleeter`)
- `spleeter_accompaniment` = path to accompaniment ONNX (required for `model_type=spleeter`)
- `threads` = CPU threads (default `2`, max `4`)
- `chunk_size` = samples per inference pass (default `44100`)

## 7) FFmpeg CLI usage examples

### UVR model

```bash
ffmpeg -i input.mp4 \
  -af "musicdamper=model_type=uvr:uvr_model=/models/UVR-MDX-NET-Voc_FT.onnx:threads=2:chunk_size=44100" \
  -c:v copy -c:a aac output-vocals.mp4
```

### Spleeter 2-stem model

```bash
ffmpeg -i input.mp4 \
  -af "musicdamper=model_type=spleeter:spleeter_vocals=/models/vocals.onnx:spleeter_accompaniment=/models/accompaniment.onnx:threads=2:chunk_size=44100" \
  -c:v copy -c:a aac output-vocals.mp4
```

## 8) mpv usage example

```bash
mpv --af='musicdamper=model_type=uvr:uvr_model=/models/UVR-MDX-NET-Voc_FT.onnx:threads=2:chunk_size=44100' video.mkv
```

## 9) AV sync and latency behavior

- The filter buffers input in `AVAudioFifo` until `chunk_size` samples are available.
- Output PTS for each processed chunk is set to the PTS of that chunk's **first input sample**.
- This preserves timeline correctness so video players can compensate predictably.
- Because Sherpa-ONNX source separation is offline/chunk-based, minimum algorithmic latency is at least one chunk.

## 10) Fallback behavior

- If model initialization fails, the filter logs an error and enters passthrough mode.
- If per-chunk inference fails at runtime, that chunk is emitted as original input audio (passthrough for that chunk), and playback continues.
