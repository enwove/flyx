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

**FLYX** ist eine lokale LLM-Inferenz-Engine ohne Abhängigkeiten, die in eine einzelne ~58 MB große Windows-`.exe` passt — schnelles spekulatives Dekodieren, nativer Langkontext, pro GPU abgestimmt, ganz ohne Einrichtung. Führen Sie unterstützte GGUF-Modelle direkt aus und stellen Sie sie Ihrem Browser, Skripten und Coding-Agenten über OpenAI- und Anthropic-kompatible APIs bereit.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Integriertes FLYX-Web-Dashboard mit Live-Durchsatz, Langkontext-Sitzungen, VRAM- und RAM-Auslastung sowie aktuellen Agenten-Anfragen">
  <br><sub>Integriertes Web-Dashboard: Modellverwaltung, Chat, Live-Durchsatz, Speicheraufteilung und Agenten-Einrichtung per Klick.</sub>
</p>

## Kernfunktionen

- **Bis zu 6× schnellere Generierung** — Integriertes spekulatives Dekodieren (MTP, DFlash2, DSpark und Prompt-Lookup): **565.5 tok/s** mit Qwen3.8-27B und **438.9 tok/s** mit Inkling-2-Mini auf einer RTX 5090 ([Leistung](#leistung)).
- **Bis zu 1M nativer Kontext auf einem Consumer-PC** — Führt Inkling-2-Mini mit 1M Kontext auf 16 GB VRAM + 32 GB RAM aus.
- **Pro GPU abgestimmt, keine manuellen Flags** — Liefert gemessene Kernel-Tuning-Tabellen für 16 NVIDIA-GPUs mit und bestimmt Kontextlänge, K/V-Präzision und GPU/RAM-Aufteilung automatisch anhand des verfügbaren Speichers. Kein Rätselraten bei Offload-Schichten, Batch-Größen oder Cache-Typen.
- **Keine Installation, keine Abhängigkeiten, ~58 MB** — CUDA- und C++-Laufzeiten sind statisch eingebunden; nur der NVIDIA-Treiber (≥ 580) wird benötigt. Kein CUDA Toolkit, Python oder Docker. Eine einzelne ~58 MB große `.exe` unter Windows (Doppelklick genügt) oder ein Tarball unter Linux.
- **Integriertes Web-Dashboard & Terminal-Monitor** — Startet direkt in einer Browser-Oberfläche (7 Sprachen) mit geführten Hinweisen zum Suchen/Herunterladen von Hugging-Face- oder lokalen Modellen, Chatten (Text & Bilder), lokalisierten API-Dokumenten sowie Live-Überwachung von Durchsatz, VRAM/RAM-Aufteilung, Sitzungen, Anfragen und Protokollen ([Monitor-Leitfaden](../monitor.md)).
- **Ein-Klick-Integration für Coding-Agenten** — Spricht OpenAI (Chat Completions, Responses) und Anthropic Messages inklusive Tool-Calls und Multi-Turn-Präfix-Wiederverwendung. Starten Sie **Claude Code** oder **Codex CLI** unter **Integration** direkt in Ihrem Projektordner (Windows) oder kopieren Sie sofort ausführbare Befehle für **DeepSeek Harness, OpenCode und Pi**.
- **Verifizierte Genauigkeit** — Jeder GPU-Kernel wird gegen eine `float64`-Referenz geprüft; standardmäßiges FP8-K/V bleibt auf 27B innerhalb von **KL 0.0009–0.0016** zu 16-Bit; spekulatives Dekodieren liefert **Token-für-Token identische** Ausgaben wie die normale Dekodierung ([Numerik](../design/numerics.md)).

## Neuigkeiten

- **`v0.2.1`** — **Auf vier GPUs verifizierte Geschwindigkeiten, Inkling-2-Mini mit DSpark und Bildeingabe**:
  - **Qwen3.8-27B + DFlash2** schreibt Code mit **565.5 tok/s** auf einer RTX 5090 (**6.03×** der normalen Dekodierung), **305.7** auf einer RTX 4090, **332.8** auf einer RTX 5070 Ti mit 16 GB und **234.1** auf einer RTX 3090. Der W4A4-Prefill mit `NVFP4` erreicht auf der 5090 **15 496 tok/s**.
  - **Inkling-2-Mini + DSpark** — ein neues Sparse-MoE-Modell, DSpark wird automatisch angehängt: **438.9 tok/s** bei Code auf der 5090 (**1.92×** der normalen Dekodierung), **322.4** auf der 4090, **276.2** auf der 5070 Ti und **222.0** auf der 3090.
  - **Vision** — Bildeingabe über die CLI, die OpenAI- und Anthropic-APIs sowie per Einfügen / Ablegen im Dashboard, für Vision-fähige Modelle mit einem `mmproj`-Projektor.
  - **Weiterhin ohne Abhängigkeiten** — eine einzelne Windows-`.exe`, jetzt ~57 MB.
- **`v0.2.0`** — **Erste Open-Source-Veröffentlichung von FLYX**: eigenständige ~59 MB Windows-`.exe` und Linux-Tarball ohne Abhängigkeiten, native Vollkontext-GGUF- & NVFP4-Inferenz mit automatischer GPU/RAM-K/V-Erweiterung, integriertes spekulatives Dekodieren (MTP / DFlash2), 7-sprachiges Web-Dashboard und Ein-Klick-Integration für Claude Code / Codex CLI.

## Leistung

Dekodiergeschwindigkeit von v0.2.1 in Token/s, auf vier GPUs verifiziert (greedy). **Normal** ist die Dekodierung ohne Drafter; die beiden Spalten **Spekulativ** schalten den Drafter für Chat- und für Code-Prompts ein, mit der Beschleunigung gegenüber Normal in Klammern.

**Qwen3.8-27B + DFlash2** — Chat / gemischt: 3 Prompts × 128 Tokens; Code: 1 Prompt × 128 Tokens.

| GPU | Quantisierung | Normal | Spekulativ · Chat / gemischt | Spekulativ · Code |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | 93.8 | **315.5** (3.36×) | **565.5** (6.03×) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | 49.4 | **171.6** (3.47×) | **305.7** (6.18×) |
| **RTX 3090** (24 GB) | `UD-Q4_K_XL` | 42.8 | **131.2** (3.06×) | **234.1** (5.46×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 68.8 | **154.4** (2.24×) | **332.8** (4.84×) |

**Inkling-2-Mini + DSpark** — Chat / gemischt: 8 Prompts × 128 Tokens; Code: 8 Prompts × 384 Tokens.

| GPU | Quantisierung | Normal | Spekulativ · Chat / gemischt | Spekulativ · Code |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `UD-IQ3_XXS` | 225.5 | **259.8** (1.15×) | **438.9** (1.92×) |
| **RTX 4090** (24 GB) | `UD-IQ3_XXS` | 166.7 | **202.9** (1.22×) | **322.4** (1.92×) |
| **RTX 3090** (24 GB) | `UD-IQ3_XXS` | 126.2 | **136.5** (1.08×) | **222.0** (1.75×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 150.9 | **169.3** (1.12×) | **276.2** (1.80×) |

<sub>**×** ist die Beschleunigung gegenüber der normalen Dekodierung mit denselben Prompts, berechnet aus ungerundeten Raten; **Normal** zeigt den Chat- / gemischten Lauf. Die spekulative Ausgabe stimmte in jedem Lauf, der Token-IDs aufzeichnete, Token für Token mit der normalen Dekodierung überein (alle außer dem gemischten Qwen-Lauf unter Windows). Linux, außer der RTX 5070 Ti unter nativem Windows 11.</sub>

<sub>**Außerdem gemessen:** Der in Qwen3.8-27B integrierte **MTP**-Drafter (keine zusätzliche Datei) erreicht **286.6 / 158.0 / 121.7 / 167.2 tok/s** mit den gemischten Prompts (5090 / 4090 / 3090 / 5070 Ti). **RTX 5090 `NVFP4`:** W4A4-Prefill mit **15 496 tok/s**; **203.8 tok/s** mit MTP bei 128K Kontext (normal: 73.4); **164.5 tok/s** bei 256K mit **100 %** Abruf (v0.2.0-Lauf). Hardware, Treiber und genaue Befehle: [docs/benchmarks.md](../benchmarks.md#v021-four-gpu-verified-benchmarks).</sub>

## Schnellstart

### 1. Herunterladen und starten (empfohlen)

Laden Sie das aktuelle Release von [GitHub Releases](https://github.com/enwove/flyx/releases) herunter — es wird lediglich ein NVIDIA-Treiber (≥ 580) benötigt:

- **Windows:** Doppelklicken Sie auf `flyx-0.2.2-windows-x86_64-cuda13.exe` (oder führen Sie `.\flyx-0.2.2-windows-x86_64-cuda13.exe serve` in PowerShell aus). Der Server startet und öffnet das Dashboard automatisch (und wählt einen anderen freien Port, falls `8080` belegt ist).
- **Linux / WSL2:** Archiv entpacken und starten:

```sh
./bin/flyx serve
# Oder ein Hugging-Face-GGUF direkt in den Standard-HF-Cache laden:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Zwei weitere Befehle zum Ausprobieren — der DSpark-Drafter und der Vision-Projektor werden mit dem Modell heruntergeladen:

```sh
# Inkling-2-Mini, DSpark wird automatisch angehängt
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
# Eine Frage zu einem Bild stellen
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
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
| Architekturen | Qwen3.5 / Qwen3.8 (`qwen35`), Qwen3 (`qwen3`), Inkling-2-Mini (`inkling2_loop`, Sparse MoE), Llama (MiniCPM5-1B) | Llama nur aus `.flyx` |
| Spekulative Drafter | MTP, DFlash2, DSpark, Prompt-Lookup | MTP und DSpark werden automatisch angehängt; DFlash2 ist eine Begleitdatei (`--drafter-file` oder **Draft with it** im Dashboard); DSpark gehört zu Inkling-2-Mini |
| Vision | Bildeingabe mit einem `mmproj`-Projektor, automatisch aus dem Hub-Repository des Modells geladen | Derzeit Qwen3.5 / Qwen3.8; Inkling-2-Mini ist nur Text |
| Modellformate | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, `MXFP4`, `Q2_0`, GGML NVFP4); `.flyx` (BF16, NVFP4; Format 2 mit eigenen LoRA-Adaptern des Zielmodells, V-LoRA, ungemergt ausgeführt; DV-Drafter nur bei expliziter Angabe `--drafter dv`, [Design](../design/dv.md)) | Alle Teile eines geteilten GGUF müssen im selben Ordner liegen. |
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
