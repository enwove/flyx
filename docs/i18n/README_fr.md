<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="../brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Licence Apache-2.0"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows et Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Rejoindre le Discord FLYX"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="QR Code du groupe WeChat"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README_zh.md">中文</a> ·
  <a href="README_ja.md">日本語</a> ·
  <a href="README_ko.md">한국어</a> ·
  <a href="README_es.md">Español</a> ·
  <b>Français</b> ·
  <a href="README_de.md">Deutsch</a>
</p>

---

<p align="center">
  <a href="#démarrage-rapide">Démarrage rapide</a> ·
  <a href="#performances">Performances</a> ·
  <a href="../models.md">Modèles</a> ·
  <a href="../server.md">CLI et API</a> ·
  <a href="../../llms.txt">llms.txt</a>
</p>

**FLYX** est un moteur d'inférence LLM local sans dépendance dans un binaire de ~50 Mo — contexte long natif, optimisé par GPU, aucune configuration requise. Exécutez vos modèles GGUF tels quels et servez-les à votre navigateur, vos scripts et vos agents de code via des APIs compatibles OpenAI et Anthropic.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Tableau de bord web intégré de FLYX affichant le débit en direct, les sessions à contexte long, l'utilisation VRAM et RAM ainsi que les requêtes récentes d'agents">
  <br><sub>Tableau de bord web intégré : gestion des modèles, chat, débit en direct, répartition mémoire et intégration d'agents en un clic.</sub>
</p>

## Fonctionnalités clés

- **Contexte long natif, sans perte de qualité** — Offre à chaque modèle l'intégralité de son contexte d'entraînement sur un seul GPU grand public. Lorsque le cache K/V dépasse la VRAM, il s'étend proprement en RAM système au lieu de tronquer ou d'appliquer une fenêtre glissante : **123 tok/s à 256K sur une RTX 4090 24 Go**, **165 tok/s à 256K sur une RTX 5090** (`NVFP4`) et **100 % de rappel en contexte long de 32K–256K** ([détails](../benchmarks.md#1-qwen38-27b-long-context-coding-benchmark)).
- **Optimisé par GPU, zéro option manuelle** — Embarque des tables d'optimisation de kernels mesurées pour 11 GPUs NVIDIA et ajuste automatiquement la longueur de contexte, la précision K/V et la répartition GPU/RAM selon la mémoire disponible. Plus besoin de deviner le nombre de couches déchargées, la taille de lot ou le type de cache.
- **Sans installation, sans dépendance, ~50 Mo** — Les runtimes CUDA et C++ sont liés statiquement ; seul le pilote NVIDIA (≥ 580) est requis. Aucun CUDA Toolkit, Python ou Docker nécessaire. Un seul `.exe` de ~50 Mo sous Windows (double-clic pour lancer), une archive de ~45 Mo sous Linux.
- **Tableau de bord web intégré et moniteur terminal** — S'ouvre directement sur une interface web (7 langues) avec des indications guidées pour rechercher/télécharger des modèles Hugging Face ou locaux, discuter, lire la documentation API localisée et suivre en direct le débit, la répartition VRAM/RAM, les sessions, les requêtes et les journaux ([guide du moniteur](../monitor.md)).
- **Intégration en un clic avec les agents de code** — Prend en charge OpenAI (Chat Completions, Responses) et Anthropic Messages avec appels d'outils et réutilisation multi-tours du préfixe. Lancez **Claude Code** ou **Codex CLI** directement dans votre dossier de projet depuis **Intégration** (Windows), ou copiez les commandes prêtes à l'emploi pour **DeepSeek Harness, OpenCode et Pi**.
- **Précision vérifiée** — Chaque kernel GPU est testé par rapport à une référence `float64` ; le cache K/V FP8 par défaut reste entre **KL 0.0009–0.0016** par rapport au 16-bit sur 27B ; chaque optimisation de vitesse garantit une sortie **strictement identique token par token** ([numérique](../design/numerics.md)).

## Actualités

- **`2026-11`** — **Première version open-source de FLYX** : `.exe` autonome de ~50 Mo pour Windows et archive tar.gz pour Linux sans dépendance, inférence GGUF et NVFP4 à contexte complet natif avec extension K/V GPU/RAM automatique, décodage spéculatif intégré (MTP / DFlash2), tableau de bord web en 7 langues et intégration en un clic pour Claude Code / Codex CLI.

## Performances

Benchmark de code greedy sur **Qwen3.8-27B GGUF** avec décodage spéculatif **MTP** intégré (`--drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos`), mesurant la génération de 512 tokens de code à **1K**, **128K** et **256K** de contexte (`NVFP4` sur RTX 5090, `UD-Q4_K_XL` sur GPUs 24 Go, `UD-IQ3_XXS` sur GPUs 16 Go ; temps exacts de prefill, placement mémoire automatique à 262K, tableaux de drafters `flyx bench` et commandes reproductibles dans [docs/benchmarks.md](../benchmarks.md)) :

| GPU | Modèle | Décodage 1K | Décodage 128K | Décodage 256K | Rappel 256K |
|---|---|---:|---:|---:|:---:|
| **RTX 5090** (32 Go) | `NVFP4` | **350.2 tok/s** | **228.7 tok/s** | **164.5 tok/s** | **100%** (4/4 multi-aiguilles) |
| **RTX 5080** (16 Go) | `UD-IQ3_XXS` | **225.9 tok/s** | **161.0 tok/s** | **105.8 tok/s** | **100%** (5/5 grille de profondeur) |
| **RTX 5070 Ti** (16 Go, WSL2) | `UD-IQ3_XXS` | **167.4 tok/s** | **104.6 tok/s** | **61.2 tok/s** | **100%** (5/5 + 4/4 multi) |
| **RTX 4090** (24 Go) | `UD-Q4_K_XL` | **193.6 tok/s** | **155.9 tok/s** | **122.7 tok/s** | **100%** (20/20 + 4/4 multi) |
| **RTX 4080** (16 Go) | `UD-IQ3_XXS` | **172.3 tok/s** | **111.0 tok/s** | **99.0 tok/s** | **100%** (3/3 @ 1K–256K) |
| **RTX 3090** (24 Go, 300 W) | `UD-Q4_K_XL` | **135.8 tok/s** | **97.5 tok/s** | **68.1 tok/s** | **100%** (3/3 @ 1K–256K) |

<sub>**RTX 5090 + DFlash2 (`NVFP4`, `--drafter all`) :** **365.0 tok/s** (1K) et **273.1 tok/s** (128K, `30.1 s` de prefill) ; les prompts mixtes `flyx bench` atteignent **311.6 tok/s** (`DFlash2`). **Rappel en contexte long (Needle-in-a-Haystack) :** **51/51 (100 %)** de rappel exact sur l'ensemble des 6 GPUs (grilles de profondeur `5%–95%` de 32K–256K et rappel 4 aiguilles à 256K, couvrant FP8 complet sur GPU et Host-RAM + StrataFlux 2 bits ; [grille complète](../benchmarks.md#3-long-context-attention-fidelity-needle-in-a-haystack-suite)).</sub>

## Démarrage rapide

### 1. Télécharger et exécuter (recommandé)

Téléchargez la dernière version sur [GitHub Releases](https://github.com/enwove/flyx/releases) — seul un pilote NVIDIA (≥ 580) est requis :

- **Windows :** Double-cliquez sur `flyx-*-windows-x86_64-cuda13.exe` (ou exécutez `.\flyx.exe serve` dans PowerShell). Le serveur démarre et ouvre automatiquement le tableau de bord (en choisissant un autre port libre si `8080` est occupé).
- **Linux / WSL2 :** Extrayez l'archive et lancez :

```sh
./bin/flyx serve
# Ou chargez directement un GGUF Hugging Face dans le cache HF standard :
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Ouvrez le tableau de bord sur `http://127.0.0.1:8080` → sélectionnez ou téléchargez un modèle dans **Modèle** → lancez **Claude Code / Codex** directement (ou copiez la commande d'un autre agent) depuis **Intégration**.

| Client / Point de terminaison | URL de base |
|---|---|
| SDKs OpenAI (Chat Completions, Responses, Completions) | `http://127.0.0.1:8080/v1` |
| SDKs Anthropic et Claude Code (`ANTHROPIC_BASE_URL`) | `http://127.0.0.1:8080` |
| Tableau de bord web | `http://127.0.0.1:8080/` |
| Télémétrie JSON (`flyx monitor`) | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>Compiler depuis les sources</b> (CUDA 13, CMake 3.24+, Ninja, g++ 11+ / MSVC 2022)</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # binaires dans ~/flyx-build/bin

# Exécuter ou servir directement par nom Hugging Face
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "Écris un haïku sur les GPUs."
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **Note WSL2 :** Conservez les répertoires de compilation et les fichiers modèles sur le système de fichiers Linux (`~/models`, `~/flyx-build`) ; `/mnt/<drive>` dégrade fortement les temps de chargement.

</details>

<details>
<summary><b>Utilisation Python et exemple curl</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Bonjour !"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "Quelle est la capitale de la France ?"}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## Modèles et matériel pris en charge

| Dimension | Pris en charge | Notes |
|---|---|---|
| Architectures | Qwen3.5 / Qwen3.8 (GDN hybride + attention à portes), Qwen3 dense, Llama (MiniCPM5-1B) | Modèles denses texte uniquement (pas de MoE ni de vision) ; Llama uniquement au format `.flyx` |
| Formats de modèle | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, GGML NVFP4) ; `.flyx` (BF16, NVFP4 ; format 2 avec adaptateurs LoRA propres à la cible, V-LoRA, exécutés sans fusion ; drafter DV activé uniquement via `--drafter dv`, [conception](../design/dv.md)) | MXFP4 refusé ; les GGUF fractionnés doivent réunir toutes les parties dans un même dossier |
| GPUs | Séries RTX 30 / 40 / 50, L4, L40S, A100, H100 (`sm_80`–`sm_120` ; `sm_100` dans la version Linux) | Un GPU par processus (`CUDA_VISIBLE_DEVICES`) ; pilote ≥ 580 |
| Plateformes | Windows (MSVC natif), Linux, WSL2 | `.exe` unique / ZIP / wheel sous Windows ; tar.gz / wheel `manylinux` sous Linux |

## Documentation

Pour les agents IA, commencez par [llms.txt](../../llms.txt).

- **Utilisation et intégration :** [docs/server.md](../server.md) · [docs/models.md](../models.md) · [docs/monitor.md](../monitor.md) · [docs/benchmarks.md](../benchmarks.md)
- **Compilation et tests :** [docs/building.md](../building.md) · [docs/testing.md](../testing.md)
- **Conception et contrats :** [docs/design/architecture.md](../design/architecture.md) · [docs/design/numerics.md](../design/numerics.md) · [docs/design/gguf.md](../design/gguf.md) · [docs/design/tree-verification.md](../design/tree-verification.md) · [docs/design/dv.md](../design/dv.md)
- **Contribution et gouvernance :** [AGENTS.md](../../AGENTS.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [SECURITY.md](../../SECURITY.md) · [docs/prior-art.md](../prior-art.md)

## Nous contacter

- **Discord :** [Rejoindre le Discord FLYX](https://discord.gg/ZtUPXBxYhN)
- **WeChat :** [Scanner le QR code du groupe WeChat](https://enwove.ai/flyx/wechatqr)
- **Problèmes et retours :** [GitHub Issues](https://github.com/enwove/flyx/issues)

Pour toute collaboration ou partenariat, contactez-nous à [contact@enwove.ai](mailto:contact@enwove.ai).

## Licence

Apache-2.0. Les attributions tierces figurent dans [NOTICE](../../NOTICE) et [docs/prior-art.md](../prior-art.md).
