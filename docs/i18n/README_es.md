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

**FLYX** es un motor de inferencia local sin dependencias. Ejecuta directamente los modelos GGUF compatibles y ofrece APIs compatibles con OpenAI y Anthropic para navegadores, scripts y agentes de programación.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Panel web integrado de FLYX mostrando rendimiento en tiempo real, sesiones de contexto largo, uso de VRAM y RAM, y solicitudes recientes de agentes">
  <br><sub>Panel web integrado: gestión de modelos, chat, rendimiento en vivo, desglose de memoria y configuración de agentes en un clic.</sub>
</p>

## Características principales

- El contexto largo aprovecha la VRAM y la RAM. Si basta con ampliar uno de estos presupuestos, FLYX indica cuántos bytes adicionales necesita la ventana solicitada.
- **Ajustado por GPU, cero flags manuales** — Incluye tablas de ajuste de kernels medidas para 11 GPUs NVIDIA y calcula automáticamente la longitud de contexto, la precisión K/V y la distribución entre GPU y RAM según la memoria disponible. Sin adivinar capas de descarga, tamaños de lote ni tipos de caché.
- **Sin instalación, sin dependencias, ~57 MB** — El ejecutable de Windows incluye los entornos de ejecución de CUDA y C++.
- **Panel web integrado y monitor de terminal** — Se abre directamente en una interfaz web (7 idiomas) con sugerencias guiadas para buscar y descargar modelos de Hugging Face o locales, chatear, consultar documentación de API localizada y supervisar en tiempo real el rendimiento, el uso de VRAM/RAM, sesiones, solicitudes y registros ([guía del monitor](../monitor.md)).
- **Integración en un clic con agentes de programación** — Compatible con OpenAI (Chat Completions, Responses) y Anthropic Messages con llamadas a herramientas y reutilización de prefijos en múltiples turnos. Lanza **Claude Code** o **Codex CLI** directamente en tu carpeta de proyecto desde **Integración** (Windows), o copia comandos listos para ejecutar para **DeepSeek Harness, OpenCode y Pi**.
- Las pruebas comparan los kernels CUDA con referencias float64. Los resultados detallan en qué equipos y cargas la decodificación especulativa produjo los mismos tokens que la normal. [float64](../design/numerics.md).

## Novedades

- **v0.2.1** — Inkling-2-Mini incorpora MoE disperso, kernels de decodificación optimizados y su drafter Markov DSpark, que se carga automáticamente. Qwen3.5/3.8 admite imágenes con un proyector compatible. El Hub descubre los archivos asociados y **Draft with it** permite elegir el drafter. La carga se adapta a la VRAM disponible. DFlash2 usa una caché KV circular para la ventana deslizante. El candidato de Windows sin PTX ocupa 57,14 MB frente a los 58,96 MB de v0.2.0.
- **`v0.2.0`** — **Primer lanzamiento open-source de FLYX**: `.exe` independiente de ~59 MB para Windows y archivo tar.gz para Linux sin dependencias, inferencia GGUF y NVFP4 de contexto completo nativo con extensión automática de K/V entre GPU y RAM, decodificación especulativa integrada (MTP / DFlash2), panel web en 7 idiomas e integración en un clic con Claude Code / Codex CLI.

## Rendimiento

Decodificación de Qwen3.8-27B en tokens/s. Las tres primeras tasas corresponden a tres prompts mixtos de 128 tokens de salida cada uno. La última usa el primer prompt de código con 128 tokens y sin búsqueda en el prompt.

| GPU | Qwen3.8-27B | Sin drafter | MTP | DFlash2 | DFlash2 en código |
|---|---|---:|---:|---:|---:|
| RTX 5090 · Linux | NVFP4 | 93.8 | 286.6 | 315.5 | 565.5 |
| RTX 4090 · Linux | UD-Q4_K_XL | 49.4 | 158.0 | 171.6 | 305.7 |
| RTX 3090 · Linux | UD-Q4_K_XL | 42.8 | 121.7 | 131.2 | 234.1 |
| RTX 5070 Ti · Windows | UD-IQ3_XXS | 68.8 | 167.2 | 154.4 | 332.8 |

Inkling-2-Mini (UD-IQ3_XXS), sin drafter → Markov DSpark (Q8_0), en tokens/s. Cada par produjo los mismos identificadores de salida.

| GPU | chat_short · 8 × 128 | code_short · 8 × 384 |
|---|---:|---:|
| RTX 5090 · Linux | 225.5 → 259.8 | 228.5 → 438.9 |
| RTX 4090 · Linux | 166.7 → 202.9 | 167.9 → 322.4 |
| RTX 3090 · Linux | 126.2 → 136.5 | 126.9 → 222.0 |
| RTX 5070 Ti · Windows | 150.9 → 169.3 | 153.6 → 276.2 |

[Mediciones y comandos](../benchmarks.md#v021-four-gpu-verified-benchmarks)

## Inicio rápido

### 1. Descargar y ejecutar (recomendado)

Descarga la última versión desde [GitHub Releases](https://github.com/enwove/flyx/releases) — solo necesitas un controlador NVIDIA (≥ 580):

- **Windows:** Haz doble clic en `flyx-0.2.1-windows-x86_64-cuda13.exe` (o ejecuta `.\flyx-0.2.1-windows-x86_64-cuda13.exe serve` en PowerShell). El servidor arranca y abre el panel web automáticamente (eligiendo otro puerto libre si `8080` está ocupado).
- **Linux / WSL2:** Extrae el archivo y ejecuta:

```sh
./bin/flyx serve
# O carga un GGUF de Hugging Face directamente en la caché estándar de HF:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Las descargas del Hub en Linux utilizan libcurl del sistema.

El primer comando carga DSpark junto al modelo. El segundo procesa una imagen con Qwen y el proyector del Hub.

```sh
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
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
| Arquitecturas | GGUF: Qwen3 (`qwen3`), Qwen3.5/3.8 (`qwen35`), Inkling-2-Mini (`inkling2_loop`) | Qwen3.5/3.8 + `qwen3vl_merger` · Inkling-2-Mini + Markov DSpark · DFlash2 / DSpark: `dflash` · Llama: `.flyx` |
| Formatos de modelo | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, `MXFP4`, `Q2_0`, GGML NVFP4); `.flyx` (BF16, NVFP4; formato 2 con adaptadores LoRA propios del objetivo, V-LoRA, sin fusionar; drafter DV solo al indicarlo explícitamente con `--drafter dv`, [diseño](../design/dv.md)) | Guarda todas las partes de un GGUF dividido en la misma carpeta. |
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
