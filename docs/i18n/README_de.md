<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="../brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Apache-2.0-Lizenz"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows und Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="FLYX-Discord beitreten"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat-Gruppen-QR-Code"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README_zh.md">中文</a> ·
  <a href="README_ja.md">日本語</a> ·
  <a href="README_ko.md">한국어</a> ·
  <a href="README_es.md">Español</a> ·
  <a href="README_fr.md">Français</a> ·
  <b>Deutsch</b>
</p>

---

<p align="center">
  <a href="#schnellstart">Schnellstart</a> ·
  <a href="#leistung">Leistung</a> ·
  <a href="../models.md">Modelle</a> ·
  <a href="../server.md">CLI &amp; API</a> ·
  <a href="../../llms.txt">llms.txt</a>
</p>

**FLYX** ist eine lokale LLM-Inferenz-Engine ohne Abhängigkeiten in einer ~50 MB großen Binärdatei — nativer Langkontext, pro GPU abgestimmt, ganz ohne Einrichtung. Führen Sie GGUF-Modelle direkt aus und stellen Sie sie Ihrem Browser, Skripten und Coding-Agenten über OpenAI- und Anthropic-kompatible APIs bereit.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Integriertes FLYX-Web-Dashboard mit Live-Durchsatz, Langkontext-Sitzungen, VRAM- und RAM-Auslastung sowie aktuellen Agenten-Anfragen">
  <br><sub>Integriertes Web-Dashboard: Modellverwaltung, Chat, Live-Durchsatz, Speicheraufteilung und Agenten-Einrichtung per Klick.</sub>
</p>

## Kernfunktionen

- **Nativer Langkontext ohne Qualitätsverlust** — Gibt jedem Modell seinen vollen trainierten Kontext auf einer einzelnen Consumer-GPU. Wenn der K/V-Cache das VRAM übersteigt, erweitert er sauber in den System-RAM statt abzuschneiden oder ein Sliding Window zu nutzen: **123 tok/s bei 256K auf einer 24 GB RTX 4090**, **165 tok/s bei 256K auf einer RTX 5090** (`NVFP4`) und **100 % Langkontext-Abruf über 32K–256K** ([Details](../benchmarks.md#1-qwen38-27b-long-context-coding-benchmark)).
- **Pro GPU abgestimmt, keine manuellen Flags** — Liefert gemessene Kernel-Tuning-Tabellen für 11 NVIDIA-GPUs mit und bestimmt Kontextlänge, K/V-Präzision und GPU/RAM-Aufteilung automatisch anhand des verfügbaren Speichers. Kein Rätselraten bei Offload-Schichten, Batch-Größen oder Cache-Typen.
- **Keine Installation, keine Abhängigkeiten, ~50 MB** — CUDA- und C++-Laufzeiten sind statisch eingebunden; nur der NVIDIA-Treiber (≥ 580) wird benötigt. Kein CUDA Toolkit, Python oder Docker. Eine einzelne ~50 MB große `.exe` unter Windows (Doppelklick genügt), ~45 MB Tarball unter Linux.
- **Integriertes Web-Dashboard & Terminal-Monitor** — Startet direkt in einer Browser-Oberfläche (7 Sprachen) mit geführten Hinweisen zum Suchen/Herunterladen von Hugging-Face- oder lokalen Modellen, Chatten, lokalisierten API-Dokumenten sowie Live-Überwachung von Durchsatz, VRAM/RAM-Aufteilung, Sitzungen, Anfragen und Protokollen ([Monitor-Leitfaden](../monitor.md)).
- **Ein-Klick-Integration für Coding-Agenten** — Spricht OpenAI (Chat Completions, Responses) und Anthropic Messages inklusive Tool-Calls und Multi-Turn-Präfix-Wiederverwendung. Starten Sie **Claude Code** oder **Codex CLI** unter **Integration** direkt in Ihrem Projektordner (Windows) oder kopieren Sie sofort ausführbare Befehle für **DeepSeek Harness, OpenCode und Pi**.
- **Verifizierte Genauigkeit** — Jeder GPU-Kernel wird gegen eine `float64`-Referenz geprüft; standardmäßiges FP8-K/V bleibt auf 27B innerhalb von **KL 0.0009–0.0016** zu 16-Bit; jede Geschwindigkeitsoptimierung liefert **Token-für-Token identische** Ausgaben ([Numerik](../design/numerics.md)).

## Neuigkeiten

- **`2026-11`** — **Erste Open-Source-Veröffentlichung von FLYX**: eigenständige ~50 MB Windows-`.exe` und Linux-Tarball ohne Abhängigkeiten, native Vollkontext-GGUF- & NVFP4-Inferenz mit automatischer GPU/RAM-K/V-Erweiterung, integriertes spekulatives Dekodieren (MTP / DFlash2), 7-sprachiges Web-Dashboard und Ein-Klick-Integration für Claude Code / Codex CLI.

## Leistung

Greedy-Coding-Benchmark auf **Qwen3.8-27B GGUF** mit integriertem spekulativem **MTP**-Dekodieren (`--drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos`), gemessen als 512-Token-Codegenerierung bei **1K**, **128K** und **256K** Kontext (`NVFP4` auf RTX 5090, `UD-Q4_K_XL` auf 24-GB-GPUs, `UD-IQ3_XXS` auf 16-GB-GPUs; exakte Prefill-Zeiten, automatische 262K-Speicheraufteilung, `flyx bench`-Drafter-Tabellen und reproduzierbare Befehle in [docs/benchmarks.md](../benchmarks.md)):

| GPU | Modell | 1K Decode | 128K Decode | 256K Decode | 256K Abruf |
|---|---|---:|---:|---:|:---:|
| **RTX 5090** (32 GB) | `NVFP4` | **350.2 tok/s** | **228.7 tok/s** | **164.5 tok/s** | **100%** (4/4 Multi-Needle) |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` | **225.9 tok/s** | **161.0 tok/s** | **105.8 tok/s** | **100%** (5/5 Tiefenraster) |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` | **167.4 tok/s** | **104.6 tok/s** | **61.2 tok/s** | **100%** (5/5 + 4/4 Multi) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | **193.6 tok/s** | **155.9 tok/s** | **122.7 tok/s** | **100%** (20/20 + 4/4 Multi) |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` | **172.3 tok/s** | **111.0 tok/s** | **99.0 tok/s** | **100%** (3/3 @ 1K–256K) |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` | **135.8 tok/s** | **97.5 tok/s** | **68.1 tok/s** | **100%** (3/3 @ 1K–256K) |

<sub>**RTX 5090 + DFlash2 (`NVFP4`, `--drafter all`):** **365.0 tok/s** (1K) und **273.1 tok/s** (128K, `30.1 s` Prefill); gemischte `flyx bench`-Prompts erreichen **311.6 tok/s** (`DFlash2`). **Langkontext-Abruf (Needle-in-a-Haystack):** **51/51 (100 %)** exakter Abruf über alle 6 GPUs (32K–256K `5%–95%`-Tiefenraster und 256K 4-Needle-Abruf, sowohl mit vollem GPU-FP8 als auch mit Host-RAM + 2-Bit-StrataFlux; [vollständige Matrix](../benchmarks.md#3-long-context-attention-fidelity-needle-in-a-haystack-suite)).</sub>

## Schnellstart

### 1. Herunterladen und starten (empfohlen)

Laden Sie das aktuelle Release von [GitHub Releases](https://github.com/enwove/flyx/releases) herunter — es wird lediglich ein NVIDIA-Treiber (≥ 580) benötigt:

- **Windows:** Doppelklicken Sie auf `flyx-*-windows-x86_64-cuda13.exe` (oder führen Sie `.\flyx.exe serve` in PowerShell aus). Der Server startet und öffnet das Dashboard automatisch (und wählt einen anderen freien Port, falls `8080` belegt ist).
- **Linux / WSL2:** Archiv entpacken und starten:

```sh
./bin/flyx serve
# Oder ein Hugging-Face-GGUF direkt in den Standard-HF-Cache laden:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Öffnen Sie das Dashboard unter `http://127.0.0.1:8080` → wählen oder laden Sie ein Modell unter **Modell** → starten Sie **Claude Code / Codex** direkt (oder kopieren Sie beliebige Agenten-Befehle) unter **Integration**.

| Client / Endpunkt | Basis-URL |
|---|---|
| OpenAI-SDKs (Chat Completions, Responses, Completions) | `http://127.0.0.1:8080/v1` |
| Anthropic-SDKs & Claude Code (`ANTHROPIC_BASE_URL`) | `http://127.0.0.1:8080` |
| Web-Dashboard | `http://127.0.0.1:8080/` |
| Telemetrie-JSON (`flyx monitor`) | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>Aus dem Quellcode bauen</b> (CUDA 13, CMake 3.24+, Ninja, g++ 11+ / MSVC 2022)</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # Ausgaben in ~/flyx-build/bin

# Direkt per Hugging-Face-Name ausführen oder bereitstellen
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "Schreibe ein Haiku über GPUs."
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **WSL2-Hinweis:** Belassen Sie Build-Verzeichnisse und Modelldateien im Linux-Dateisystem (`~/models`, `~/flyx-build`); `/mnt/<drive>` verlangsamt das Laden von Modellen erheblich.

</details>

<details>
<summary><b>Python-Nutzung &amp; curl-Beispiel</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hallo!"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "Wie heißt die Hauptstadt von Frankreich?"}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## Unterstützte Modelle & Hardware

| Bereich | Unterstützt | Hinweise |
|---|---|---|
| Architekturen | Qwen3.5 / Qwen3.8 (hybrides GDN + Gated Attention), Qwen3 Dense, Llama (MiniCPM5-1B) | Nur reine Text-Dense-Modelle (kein MoE, kein Vision); Llama nur aus `.flyx` |
| Modellformate | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, GGML NVFP4); `.flyx` (BF16, NVFP4; Format 2 mit eigenen LoRA-Adaptern des Zielmodells, V-LoRA, ungemergt ausgeführt; DV-Drafter nur bei expliziter Angabe `--drafter dv`, [Design](../design/dv.md)) | MXFP4 wird abgelehnt; geteilte GGUFs müssen alle Teile im selben Ordner haben |
| GPUs | RTX-30-/40-/50-Serie, L4, L40S, A100, H100 (`sm_80`–`sm_120`; `sm_100` im Linux-Release) | Eine GPU pro Prozess (`CUDA_VISIBLE_DEVICES`); Treiber ≥ 580 |
| Plattformen | Windows (natives MSVC), Linux, WSL2 | Einzelne `.exe` / ZIP / Wheel unter Windows; Tarball / `manylinux`-Wheel unter Linux |

## Dokumentation

Für KI-Agenten empfiehlt sich der Einstieg über [llms.txt](../../llms.txt).

- **Nutzung & Integration:** [docs/server.md](../server.md) · [docs/models.md](../models.md) · [docs/monitor.md](../monitor.md) · [docs/benchmarks.md](../benchmarks.md)
- **Build & Tests:** [docs/building.md](../building.md) · [docs/testing.md](../testing.md)
- **Design & Verträge:** [docs/design/architecture.md](../design/architecture.md) · [docs/design/numerics.md](../design/numerics.md) · [docs/design/gguf.md](../design/gguf.md) · [docs/design/tree-verification.md](../design/tree-verification.md) · [docs/design/dv.md](../design/dv.md)
- **Mitwirken & Governance:** [AGENTS.md](../../AGENTS.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [SECURITY.md](../../SECURITY.md) · [docs/prior-art.md](../prior-art.md)

## Kontakt

- **Discord:** [Dem FLYX-Discord beitreten](https://discord.gg/ZtUPXBxYhN)
- **WeChat:** [QR-Code der WeChat-Gruppe scannen](https://enwove.ai/flyx/wechatqr)
- **Fehler & Feedback:** [GitHub Issues](https://github.com/enwove/flyx/issues)

Für Kooperationen und Partnerschaften erreichen Sie uns unter [contact@enwove.ai](mailto:contact@enwove.ai).

## Lizenz

Apache-2.0. Hinweise zu Drittanbietern finden sich in [NOTICE](../../NOTICE) und [docs/prior-art.md](../prior-art.md).
