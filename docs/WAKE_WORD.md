# Training the "MIKATA" wake word

Without a trained model, MIKATA reacts to **"Hey Jarvis"** (openWakeWord model installed by
`download_models.py`) and to the **Ctrl+Alt+M** shortcut. This guide produces two files, tuned
to your voice and your microphone:

- `models/mikata.onnx`: the wake word detector, which outputs a score every 80 ms;
- `models/mikata_verifier.pkl`: the verifier, which gives the probability that **you** are the
  one speaking. A "MIKATA" said by a video or by someone else is rejected.

Plan for about 4 hours of passive recording (spread over several days), 20 minutes of
recording yourself and 2 to 3 hours of training on Google Colab (free GPU). All commands run
in PowerShell, from the repository folder. The scripts' messages are in French.

## 1. Record your everyday background (at least 4 h)

This is the most valuable data: it teaches the model what is **not** "MIKATA". In
`config.yaml`, set `wake.collect_negatives: true`, start MIKATA and go about your day (games,
music, videos, calls, a quiet room). Do not say "MIKATA" during the collection. It stops by
itself after `wake.negatives_max_hours` (12 h, ~115 MB per hour).

```powershell
uv run python scripts\record_wakeword_samples.py plan   # what is done, what is missing
```

Set `collect_negatives: false` again when you are done.

## 2. Record your "MIKATA"s (≈ 20 min)

With your usual microphone, at your usual spot, the way you will say it every day:

```powershell
uv run python scripts\record_wakeword_samples.py record --set train   # 100 takes
uv run python scripts\record_wakeword_samples.py record --set hard    # 36 similar words x 3
```

- `train` covers 9 situations: normal voice, low, loud, tired, smiling, from afar, head
  turned away, with background noise, and within a sentence ("MIKATA, ouvre Spotify").
- `hard` records similar-sounding words to reject (mikado, Nikita, Akira, "ma carte"...).
- Enter keeps the take, `r` redoes it, `q` quits. Running it again resumes where you stopped;
  `--only loin` redoes a single situation (situation names are in French, as listed by `plan`).

## 3. Another day: 30 evaluation takes

```powershell
uv run python scripts\record_wakeword_samples.py record --set eval
uv run python scripts\record_wakeword_samples.py plan
```

These takes are never used for training: they measure the model on "another day". `plan`
should say "tout est prêt" (all set).

## 4. Train `mikata.onnx` on Google Colab (≈ 2 to 3 h)

```powershell
uv run python scripts\record_wakeword_samples.py pack   # --negative-hours 2 for a smaller archive
```

This creates `data\wakeword\mikata_wakeword_colab.zip` (~500 MB). Then:

1. On <https://colab.research.google.com>, create a notebook, then **Runtime > Change runtime
   type > T4 GPU**.
2. Put the archive in your Google Drive (`My Drive`), then run these cells one by one:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   !cp /content/drive/MyDrive/mikata_wakeword_colab.zip /content/
   ```

   ```python
   !unzip -q -o /content/mikata_wakeword_colab.zip -d /content/mikata
   !python /content/mikata/train_wakeword.py setup   # ~15 min, ~17 GB of public data
   !python /content/mikata/train_wakeword.py train
   ```

   `train` ends with `[mikata] modèle prêt : /content/mikata/out/mikata.onnx` (a TensorFlow or
   tflite error after that line is harmless). If Colab disconnects, run the same cells again:
   steps already done are skipped.
3. Download the model and copy it into `models\`:

   ```python
   from google.colab import files
   files.download("/content/mikata/out/mikata.onnx")
   ```

## 5. Train the voice verifier (≈ 1 min, on your PC)

```powershell
uv run python scripts\train_verifier.py
```

It learns your voice saying "MIKATA" (the `train` takes) against your voice saying something
else (similar words, and your sentences in `data\stt_eval` if you used `eval_stt.py`).

## 6. Measure and pick the thresholds (≈ 10 min)

```powershell
uv run python scripts\eval_wakeword.py --model models\mikata.onnx --verifier models\mikata_verifier.pkl
```

The script replays your evaluation takes inside 2 hours of everyday background and shows, for
each threshold: false activations per hour, misses and delay. It ends with a **recommended
setting** to copy into `config.yaml`. Target: fewer than 0.2 false activations per hour and
fewer than 5 % misses. Otherwise, add takes or background audio and retrain
(`train --user-copies 50`, `--steps 80000`...).

## 7. Switch to production

In `config.yaml`:

```yaml
wake:
  model: models/mikata.onnx
  verifier: models/mikata_verifier.pkl
  threshold: 0.50              # <- recommended setting
  consecutive_frames: 1        # <- recommended setting
  verifier_threshold: 0.70     # <- recommended setting
```

You can remove `"jarvis"` from `stt.wake_words`, then:

```powershell
uv run python -m mikata --check-config
uv run python -m mikata
```

Missed a "MIKATA"? Press Ctrl+Alt+M right after: the last 4 seconds are saved in
`data\wakeword\misses\` (`wake.save_misses`), useful for a future training run.
