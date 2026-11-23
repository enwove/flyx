<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="../brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Licencia Apache-2.0"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows y Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Unirse al Discord de FLYX"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="Código QR del grupo de WeChat"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README_zh.md">中文</a> ·
  <a href="README_ja.md">日本語</a> ·
  <a href="README_ko.md">한국어</a> ·
  <b>Español</b> ·
  <a href="README_fr.md">Français</a> ·
  <a href="README_de.md">Deutsch</a>
</p>

---

<p align="center">
  <a href="#inicio-rápido">Inicio rápido</a> ·
  <a href="#rendimiento">Rendimiento</a> ·
  <a href="../models.md">Modelos</a> ·
  <a href="../server.md">CLI y API</a> ·
  <a href="../../llms.txt">llms.txt</a>
</p>

**FLYX** es un motor de inferencia local de LLM sin dependencias en un binario de ~50 MB: contexto largo nativo, optimizado por GPU y sin configuración. Ejecuta modelos GGUF tal como están y sírvelos a tu navegador, scripts y agentes de programación mediante APIs compatibles con OpenAI y Anthropic.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Panel web integrado de FLYX mostrando rendimiento en tiempo real, sesiones de contexto largo, uso de VRAM y RAM, y solicitudes recientes de agentes">
  <br><sub>Panel web integrado: gestión de modelos, chat, rendimiento en vivo, desglose de memoria y configuración de agentes en un clic.</sub>
</p>

## Características principales

- **Contexto largo nativo, sin pérdida de calidad** — Ofrece a cada modelo todo su contexto entrenado en una sola GPU de consumo. Cuando la caché K/V supera la VRAM, se extiende limpiamente a la RAM del sistema en lugar de truncar o usar ventana deslizante: **123 tok/s a 256K en una RTX 4090 de 24 GB**, **165 tok/s a 256K en una RTX 5090** (`NVFP4`) y **100 % de recuperación en contexto largo de 32K–256K** ([detalles](../benchmarks.md#1-qwen38-27b-long-context-coding-benchmark)).
- **Ajustado por GPU, cero flags manuales** — Incluye tablas de ajuste de kernels medidas para 11 GPUs NVIDIA y calcula automáticamente la longitud de contexto, la precisión K/V y la distribución entre GPU y RAM según la memoria disponible. Sin adivinar capas de descarga, tamaños de lote ni tipos de caché.
- **Sin instalación, sin dependencias, ~50 MB** — Los runtimes de CUDA y C++ están enlazados estáticamente; solo se requiere el controlador NVIDIA (≥ 580). Sin CUDA Toolkit, Python ni Docker. Un único `.exe` de ~50 MB en Windows (doble clic para ejecutar) y un archivo de ~45 MB en Linux.
- **Panel web integrado y monitor de terminal** — Se abre directamente en una interfaz web (7 idiomas) con sugerencias guiadas para buscar y descargar modelos de Hugging Face o locales, chatear, consultar documentación de API localizada y supervisar en tiempo real el rendimiento, el uso de VRAM/RAM, sesiones, solicitudes y registros ([guía del monitor](../monitor.md)).
- **Integración en un clic con agentes de programación** — Compatible con OpenAI (Chat Completions, Responses) y Anthropic Messages con llamadas a herramientas y reutilización de prefijos en múltiples turnos. Lanza **Claude Code** o **Codex CLI** directamente en tu carpeta de proyecto desde **Integración** (Windows), o copia comandos listos para ejecutar para **DeepSeek Harness, OpenCode y Pi**.
- **Precisión verificada** — Cada kernel GPU se verifica frente a una referencia en `float64`; el formato FP8 K/V predeterminado se mantiene en **KL 0.0009–0.0016** respecto a 16-bit en 27B; cada optimización de velocidad conserva una salida **idéntica token a token** ([numérica](../design/numerics.md)).

## Novedades

- **`2026-11`** — **Primer lanzamiento open-source de FLYX**: `.exe` independiente de ~50 MB para Windows y archivo tar.gz para Linux sin dependencias, inferencia GGUF y NVFP4 de contexto completo nativo con extensión automática de K/V entre GPU y RAM, decodificación especulativa integrada (MTP / DFlash2), panel web en 7 idiomas e integración en un clic con Claude Code / Codex CLI.

## Rendimiento

Benchmark de programación greedy en **Qwen3.8-27B GGUF** con decodificación especulativa **MTP** integrada (`--drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos`), midiendo la generación de 512 tokens de código a **1K**, **128K** y **256K** de contexto (`NVFP4` en RTX 5090, `UD-Q4_K_XL` en GPUs de 24 GB, `UD-IQ3_XXS` en GPUs de 16 GB; tiempos exactos de prefill, colocación automática de memoria a 262K, tablas de drafters de `flyx bench` y comandos reproducibles en [docs/benchmarks.md](../benchmarks.md)):

| GPU | Modelo | Decode 1K | Decode 128K | Decode 256K | Recuperación 256K |
|---|---|---:|---:|---:|:---:|
| **RTX 5090** (32 GB) | `NVFP4` | **350.2 tok/s** | **228.7 tok/s** | **164.5 tok/s** | **100%** (4/4 multi-aguja) |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` | **225.9 tok/s** | **161.0 tok/s** | **105.8 tok/s** | **100%** (5/5 por profundidad) |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` | **167.4 tok/s** | **104.6 tok/s** | **61.2 tok/s** | **100%** (5/5 + 4/4 multi) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | **193.6 tok/s** | **155.9 tok/s** | **122.7 tok/s** | **100%** (20/20 + 4/4 multi) |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` | **172.3 tok/s** | **111.0 tok/s** | **99.0 tok/s** | **100%** (3/3 @ 1K–256K) |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` | **135.8 tok/s** | **97.5 tok/s** | **68.1 tok/s** | **100%** (3/3 @ 1K–256K) |

<sub>**RTX 5090 + DFlash2 (`NVFP4`, `--drafter all`):** **365.0 tok/s** (1K) y **273.1 tok/s** (128K, `30.1 s` de prefill); los prompts mixtos de `flyx bench` alcanzan **311.6 tok/s** (`DFlash2`). **Recuperación en contexto largo (Needle-in-a-Haystack):** **51/51 (100 %)** de recuperación exacta en las 6 GPUs (cuadrículas de profundidad `5%–95%` de 32K–256K y recuperación de 4 agujas a 256K, tanto en FP8 completo en GPU como en Host-RAM + StrataFlux de 2 bits; [cuadrícula completa](../benchmarks.md#3-long-context-attention-fidelity-needle-in-a-haystack-suite)).</sub>

## Inicio rápido

### 1. Descargar y ejecutar (recomendado)

Descarga la última versión desde [GitHub Releases](https://github.com/enwove/flyx/releases) — solo necesitas un controlador NVIDIA (≥ 580):

- **Windows:** Haz doble clic en `flyx-*-windows-x86_64-cuda13.exe` (o ejecuta `.\flyx.exe serve` en PowerShell). El servidor arranca y abre el panel web automáticamente (eligiendo otro puerto libre si `8080` está ocupado).
- **Linux / WSL2:** Extrae el archivo y ejecuta:

```sh
./bin/flyx serve
# O carga un GGUF de Hugging Face directamente en la caché estándar de HF:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Abre el panel en `http://127.0.0.1:8080` → elige o descarga un modelo en **Modelo** → lanza **Claude Code / Codex** directamente (o copia cualquier comando de agente) desde **Integración**.

| Cliente / Endpoint | URL base |
|---|---|
| SDKs de OpenAI (Chat Completions, Responses, Completions) | `http://127.0.0.1:8080/v1` |
| SDKs de Anthropic y Claude Code (`ANTHROPIC_BASE_URL`) | `http://127.0.0.1:8080` |
| Panel web | `http://127.0.0.1:8080/` |
| Telemetría JSON (`flyx monitor`) | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>Compilar desde el código fuente</b> (CUDA 13, CMake 3.24+, Ninja, g++ 11+ / MSVC 2022)</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # binarios en ~/flyx-build/bin

# Ejecutar o servir directamente por nombre de Hugging Face
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "Escribe un haiku sobre GPUs."
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **Nota sobre WSL2:** Mantén los directorios de compilación y los archivos de modelos en el sistema de archivos de Linux (`~/models`, `~/flyx-build`); usar `/mnt/<drive>` degrada gravemente los tiempos de carga.

</details>

<details>
<summary><b>Uso en Python y ejemplo con curl</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "¡Hola!"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "¿Cuál es la capital de Francia?"}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## Modelos y hardware compatibles

| Dimensión | Compatible | Notas |
|---|---|---|
| Arquitecturas | Qwen3.5 / Qwen3.8 (GDN híbrido + atención con compuerta), Qwen3 denso, Llama (MiniCPM5-1B) | Solo modelos densos de texto (sin MoE ni visión); Llama solo desde `.flyx` |
| Formatos de modelo | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, GGML NVFP4); `.flyx` (BF16, NVFP4; formato 2 con adaptadores LoRA propios del objetivo, V-LoRA, sin fusionar; drafter DV solo al indicarlo explícitamente con `--drafter dv`, [diseño](../design/dv.md)) | MXFP4 no admitido; los GGUF divididos requieren todas las partes en la misma carpeta |
| GPUs | Series RTX 30 / 40 / 50, L4, L40S, A100, H100 (`sm_80`–`sm_120`; `sm_100` en la versión de Linux) | Una GPU por proceso (`CUDA_VISIBLE_DEVICES`); controlador ≥ 580 |
| Plataformas | Windows (MSVC nativo), Linux, WSL2 | `.exe` único / ZIP / wheel en Windows; tar.gz / wheel `manylinux` en Linux |

## Documentación

Para agentes de IA, comienza por [llms.txt](../../llms.txt).

- **Uso e integración:** [docs/server.md](../server.md) · [docs/models.md](../models.md) · [docs/monitor.md](../monitor.md) · [docs/benchmarks.md](../benchmarks.md)
- **Compilación y pruebas:** [docs/building.md](../building.md) · [docs/testing.md](../testing.md)
- **Diseño y contratos:** [docs/design/architecture.md](../design/architecture.md) · [docs/design/numerics.md](../design/numerics.md) · [docs/design/gguf.md](../design/gguf.md) · [docs/design/tree-verification.md](../design/tree-verification.md) · [docs/design/dv.md](../design/dv.md)
- **Contribución y gobernanza:** [AGENTS.md](../../AGENTS.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [SECURITY.md](../../SECURITY.md) · [docs/prior-art.md](../prior-art.md)

## Contacto

- **Discord:** [Únete al Discord de FLYX](https://discord.gg/ZtUPXBxYhN)
- **WeChat:** [Escanea el código QR del grupo de WeChat](https://enwove.ai/flyx/wechatqr)
- **Incidencias y comentarios:** [GitHub Issues](https://github.com/enwove/flyx/issues)

Para colaboraciones y alianzas, escríbenos a [contact@enwove.ai](mailto:contact@enwove.ai).

## Licencia

Apache-2.0. Las atribuciones de terceros figuran en [NOTICE](../../NOTICE) y [docs/prior-art.md](../prior-art.md).
