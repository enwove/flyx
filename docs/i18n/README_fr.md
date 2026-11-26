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

**FLYX** est un moteur d’inférence local sans dépendance. Il exécute directement les modèles GGUF pris en charge et les sert aux navigateurs, scripts et agents de développement via des API compatibles avec OpenAI et Anthropic.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Tableau de bord web intégré de FLYX affichant le débit en direct, les sessions à contexte long, l'utilisation VRAM et RAM ainsi que les requêtes récentes d'agents">
  <br><sub>Tableau de bord web intégré : gestion des modèles, chat, débit en direct, répartition mémoire et intégration d'agents en un clic.</sub>
</p>

## Fonctionnalités clés

- Le contexte long utilise la VRAM et la RAM. Si la fenêtre demandée tient en augmentant un seul de ces budgets, FLYX indique combien d’octets il manque.
- **Optimisé par GPU, zéro option manuelle** — Embarque des tables d'optimisation de kernels mesurées pour 11 GPUs NVIDIA et ajuste automatiquement la longueur de contexte, la précision K/V et la répartition GPU/RAM selon la mémoire disponible. Plus besoin de deviner le nombre de couches déchargées, la taille de lot ou le type de cache.
- **Sans installation, sans dépendance, ~57 Mo** — L’exécutable Windows inclut les bibliothèques d’exécution CUDA et C++.
- **Tableau de bord web intégré et moniteur terminal** — S'ouvre directement sur une interface web (7 langues) avec des indications guidées pour rechercher/télécharger des modèles Hugging Face ou locaux, discuter, lire la documentation API localisée et suivre en direct le débit, la répartition VRAM/RAM, les sessions, les requêtes et les journaux ([guide du moniteur](../monitor.md)).
- **Intégration en un clic avec les agents de code** — Prend en charge OpenAI (Chat Completions, Responses) et Anthropic Messages avec appels d'outils et réutilisation multi-tours du préfixe. Lancez **Claude Code** ou **Codex CLI** directement dans votre dossier de projet depuis **Intégration** (Windows), ou copiez les commandes prêtes à l'emploi pour **DeepSeek Harness, OpenCode et Pi**.
- Les tests comparent les noyaux CUDA à des références en float64. Les mesures précisent les cartes et les requêtes pour lesquelles le décodage spéculatif a produit les mêmes tokens que le décodage ordinaire. [float64](../design/numerics.md).

## Actualités

- **v0.2.1** — Inkling-2-Mini bénéficie d’un MoE à activation clairsemée, de noyaux de décodage optimisés et du chargement automatique de son drafter Markov DSpark. Qwen3.5/3.8 accepte les images avec un projecteur compatible. Le Hub détecte les fichiers associés et **Draft with it** permet de choisir le drafter. Le chargement s’adapte à la VRAM disponible. DFlash2 utilise un tampon KV circulaire pour la fenêtre glissante. L’exécutable Windows candidat sans PTX pèse 57,15 Mo contre 58,96 Mo pour la v0.2.0.
- **`v0.2.0`** — **Première version open-source de FLYX** : `.exe` autonome de ~59 Mo pour Windows et archive tar.gz pour Linux sans dépendance, inférence GGUF et NVFP4 à contexte complet natif avec extension K/V GPU/RAM automatique, décodage spéculatif intégré (MTP / DFlash2), tableau de bord web en 7 langues et intégration en un clic pour Claude Code / Codex CLI.

## Performances

Décodage de Qwen3.8-27B en tokens/s. Les trois premiers débits portent sur trois requêtes mixtes de 128 tokens de sortie chacune. Le dernier utilise la première requête de code, avec 128 tokens et sans recherche dans le prompt.

| GPU | Qwen3.8-27B | Sans drafter | MTP | DFlash2 | DFlash2 sur code |
|---|---|---:|---:|---:|---:|
| RTX 5090 · Linux | NVFP4 | 93.8 | 286.6 | 315.5 | 565.5 |
| RTX 4090 · Linux | UD-Q4_K_XL | 49.4 | 158.0 | 171.6 | 305.7 |
| RTX 3090 · Linux | UD-Q4_K_XL | 42.8 | 121.7 | 131.2 | 234.1 |
| RTX 5070 Ti · Windows | UD-IQ3_XXS | 68.8 | 167.2 | 154.4 | 332.8 |

Inkling-2-Mini (UD-IQ3_XXS), sans drafter → Markov DSpark (Q8_0), en tokens/s. Les identifiants de sortie sont identiques pour chaque paire.

| GPU | chat_short · 8 × 128 | code_short · 8 × 384 |
|---|---:|---:|
| RTX 5090 · Linux | 225.5 → 259.8 | 228.5 → 438.9 |
| RTX 5070 Ti · Windows | 150.9 → 169.3 | 153.6 → 276.2 |

[Mesures et commandes](../benchmarks.md#v021-four-gpu-verified-benchmarks)

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

Sous Linux, les téléchargements du Hub utilisent la bibliothèque libcurl du système.

La première commande charge DSpark avec le modèle. La seconde traite une image avec Qwen et le projecteur du Hub.

```sh
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
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
| Architectures | GGUF: Qwen3 (`qwen3`), Qwen3.5/3.8 (`qwen35`), Inkling-2-Mini (`inkling2_loop`) | Qwen3.5/3.8 + `qwen3vl_merger` · Inkling-2-Mini + Markov DSpark · DFlash2 / DSpark: `dflash` · Llama: `.flyx` |
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
