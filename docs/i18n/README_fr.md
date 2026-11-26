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

**FLYX** est un moteur d'inférence LLM local sans dépendance qui tient dans un seul `.exe` Windows de ~57 Mo — décodage spéculatif rapide, contexte long natif, optimisé par GPU, aucune configuration requise. Exécutez les modèles GGUF pris en charge tels quels et servez-les à votre navigateur, vos scripts et vos agents de code via des APIs compatibles OpenAI et Anthropic.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Tableau de bord web intégré de FLYX affichant le débit en direct, les sessions à contexte long, l'utilisation VRAM et RAM ainsi que les requêtes récentes d'agents">
  <br><sub>Tableau de bord web intégré : gestion des modèles, chat, débit en direct, répartition mémoire et intégration d'agents en un clic.</sub>
</p>

## Fonctionnalités clés

- **Un décodage spéculatif qui ne change jamais la sortie** — Drafters MTP, DFlash2 et DSpark intégrés, plus la recherche dans le prompt : Qwen3.8-27B écrit du code à **565.5 tok/s** sur une RTX 5090, soit **6.03×** le décodage ordinaire, et chaque token est celui que le décodage ordinaire aurait produit ([Performances](#performances)).
- **Contexte long natif, sans perte de qualité** — Exécute Qwen3.8-27B avec l'intégralité de son contexte de 262K sur un seul GPU grand public, cartes 16 Go comprises. Lorsque le cache K/V dépasse la VRAM, il s'étend proprement en RAM système au lieu de tronquer ou d'appliquer une fenêtre glissante : **164.5 tok/s à 256K sur une RTX 5090** (`NVFP4`), **122.7 tok/s à 256K sur une RTX 4090 24 Go** et **100 % de rappel en contexte long de 32K–256K** ([mesures v0.2.0](../benchmarks.md#1-qwen38-27b-long-context-coding-benchmark)).
- **Optimisé par GPU, zéro option manuelle** — Embarque des tables d'optimisation de kernels mesurées pour 11 GPUs NVIDIA et ajuste automatiquement la longueur de contexte, la précision K/V et la répartition GPU/RAM selon la mémoire disponible. Plus besoin de deviner le nombre de couches déchargées, la taille de lot ou le type de cache.
- **Sans installation, sans dépendance, ~57 Mo** — Les runtimes CUDA et C++ sont liés statiquement ; seul le pilote NVIDIA (≥ 580) est requis. Aucun CUDA Toolkit, Python ou Docker nécessaire. Un seul `.exe` de ~57 Mo sous Windows (double-clic pour lancer), ou une archive tar.gz sous Linux.
- **Entrée d'images** — Les modèles dotés de la vision et d'un projecteur `mmproj` acceptent des images depuis la CLI (`--image`), les APIs OpenAI et Anthropic, et par collage ou glisser-déposer dans le tableau de bord.
- **Tableau de bord web intégré et moniteur terminal** — S'ouvre directement sur une interface web (7 langues) avec des indications guidées pour rechercher/télécharger des modèles Hugging Face ou locaux, discuter, lire la documentation API localisée et suivre en direct le débit, la répartition VRAM/RAM, les sessions, les requêtes et les journaux ([guide du moniteur](../monitor.md)).
- **Intégration en un clic avec les agents de code** — Prend en charge OpenAI (Chat Completions, Responses) et Anthropic Messages avec appels d'outils et réutilisation multi-tours du préfixe. Lancez **Claude Code** ou **Codex CLI** directement dans votre dossier de projet depuis **Intégration** (Windows), ou copiez les commandes prêtes à l'emploi pour **DeepSeek Harness, OpenCode et Pi**.
- **Précision vérifiée** — Chaque kernel GPU est testé par rapport à une référence `float64` ; le cache K/V FP8 par défaut reste entre **KL 0.0009–0.0016** par rapport au 16-bit sur 27B ; le décodage spéculatif est **strictement identique token par token** au décodage ordinaire ([numérique](../design/numerics.md)).

## Actualités

- **`v0.2.1`** — **Vitesses vérifiées sur quatre GPUs, Inkling-2-Mini avec DSpark et entrée d'images** :
  - **Qwen3.8-27B + DFlash2** écrit du code à **565.5 tok/s** sur une RTX 5090 (**6.03×** le décodage ordinaire), **305.7** sur une RTX 4090, **332.8** sur une RTX 5070 Ti 16 Go et **234.1** sur une RTX 3090. Le prefill W4A4 en `NVFP4` sur la 5090 atteint **15 496 tok/s**.
  - **Inkling-2-Mini + DSpark** — un nouveau modèle MoE clairsemé, avec DSpark attaché automatiquement : **438.9 tok/s** en code sur la 5090 (**1.92×** le décodage ordinaire), **322.4** sur la 4090, **276.2** sur la 5070 Ti et **222.0** sur la 3090.
  - **Vision** — entrée d'images depuis la CLI, les APIs OpenAI et Anthropic, et par collage / dépôt dans le tableau de bord, pour les modèles dotés de la vision et d'un projecteur `mmproj`.
  - **Toujours sans dépendance** — un seul `.exe` Windows, désormais ~57 Mo.
- **`v0.2.0`** — **Première version open-source de FLYX** : `.exe` autonome de ~59 Mo pour Windows et archive tar.gz pour Linux sans dépendance, inférence GGUF et NVFP4 à contexte complet natif avec extension K/V GPU/RAM automatique, décodage spéculatif intégré (MTP / DFlash2), tableau de bord web en 7 langues et intégration en un clic pour Claude Code / Codex CLI.

## Performances

Vitesse de décodage de la v0.2.1 en tokens/s, vérifiée sur quatre GPUs (greedy). **Ordinaire** désigne le décodage sans drafter ; les deux colonnes **Spéculatif** activent le drafter sur des prompts de chat et de code, avec l'accélération par rapport à l'ordinaire entre parenthèses.

**Qwen3.8-27B + DFlash2** — chat / mixte : 3 prompts × 128 tokens ; code : 1 prompt × 128 tokens.

| GPU | Quantification | Ordinaire | Spéculatif · chat / mixte | Spéculatif · code |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | 93.8 | **315.5** (3.36×) | **565.5** (6.03×) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | 49.4 | **171.6** (3.47×) | **305.7** (6.18×) |
| **RTX 3090** (24 GB) | `UD-Q4_K_XL` | 42.8 | **131.2** (3.06×) | **234.1** (5.46×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 68.8 | **154.4** (2.24×) | **332.8** (4.84×) |

**Inkling-2-Mini + DSpark** — chat / mixte : 8 prompts × 128 tokens ; code : 8 prompts × 384 tokens.

| GPU | Quantification | Ordinaire | Spéculatif · chat / mixte | Spéculatif · code |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `UD-IQ3_XXS` | 225.5 | **259.8** (1.15×) | **438.9** (1.92×) |
| **RTX 4090** (24 GB) | `UD-IQ3_XXS` | 166.7 | **202.9** (1.22×) | **322.4** (1.92×) |
| **RTX 3090** (24 GB) | `UD-IQ3_XXS` | 126.2 | **136.5** (1.08×) | **222.0** (1.75×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 150.9 | **169.3** (1.12×) | **276.2** (1.80×) |

<sub>**×** est l'accélération par rapport au décodage ordinaire sur les mêmes prompts, calculée à partir des débits non arrondis ; **Ordinaire** correspond à l'exécution chat / mixte. La sortie spéculative correspond token par token au décodage ordinaire dans toutes les exécutions ayant enregistré les identifiants de tokens (toutes sauf l'exécution mixte de Qwen sous Windows). Linux, sauf la RTX 5070 Ti sous Windows 11 natif.</sub>

<sub>**Également mesuré :** le drafter **MTP** intégré à Qwen3.8-27B (sans fichier supplémentaire) atteint **286.6 / 158.0 / 121.7 / 167.2 tok/s** sur les prompts mixtes (5090 / 4090 / 3090 / 5070 Ti). **RTX 5090 `NVFP4` :** prefill W4A4 à **15 496 tok/s** ; **203.8 tok/s** avec MTP à 128K de contexte (ordinaire : 73.4) ; **164.5 tok/s** à 256K avec **100 %** de rappel (exécution v0.2.0). Matériel, pilotes et commandes exactes : [docs/benchmarks.md](../benchmarks.md#v021-four-gpu-verified-benchmarks).</sub>

## Démarrage rapide

### 1. Télécharger et exécuter (recommandé)

Téléchargez la dernière version sur [GitHub Releases](https://github.com/enwove/flyx/releases) — seul un pilote NVIDIA (≥ 580) est requis :

- **Windows :** Double-cliquez sur `flyx-0.2.1-windows-x86_64-cuda13.exe` (ou exécutez `.\flyx-0.2.1-windows-x86_64-cuda13.exe serve` dans PowerShell). Le serveur démarre et ouvre automatiquement le tableau de bord (en choisissant un autre port libre si `8080` est occupé).
- **Linux / WSL2 :** Extrayez l'archive et lancez :

```sh
./bin/flyx serve
# Ou chargez directement un GGUF Hugging Face dans le cache HF standard :
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Deux autres commandes à essayer — le drafter DSpark et le projecteur de vision se téléchargent avec le modèle :

```sh
# Inkling-2-Mini, avec DSpark attaché automatiquement
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
# Poser une question sur une image
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
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
| Architectures | Qwen3.5 / Qwen3.8 (`qwen35`), Qwen3 (`qwen3`), Inkling-2-Mini (`inkling2_loop`, MoE clairsemé), Llama (MiniCPM5-1B) | Llama uniquement depuis `.flyx` |
| Drafters spéculatifs | MTP, DFlash2, DSpark, recherche dans le prompt | MTP et DSpark s'attachent automatiquement ; DFlash2 est un fichier compagnon (`--drafter-file`, ou **Draft with it** dans le tableau de bord) ; DSpark accompagne Inkling-2-Mini |
| Vision | Entrée d'images avec un projecteur `mmproj`, récupéré automatiquement depuis le dépôt Hub du modèle | Qwen3.5 / Qwen3.8 à ce jour ; Inkling-2-Mini est en texte seul |
| Formats de modèle | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, `MXFP4`, `Q2_0`, GGML NVFP4) ; `.flyx` (BF16, NVFP4 ; format 2 avec adaptateurs LoRA propres à la cible, V-LoRA, exécutés sans fusion ; drafter DV activé uniquement via `--drafter dv`, [conception](../design/dv.md)) | Placez tous les fragments d’un GGUF dans le même dossier. |
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
