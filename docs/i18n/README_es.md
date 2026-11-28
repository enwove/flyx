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

**FLYX** es un motor de inferencia local de LLM sin dependencias que cabe en un único `.exe` de ~58 MB para Windows: decodificación especulativa rápida, contexto largo nativo, optimizado por GPU y sin configuración. Ejecuta los modelos GGUF compatibles tal como están y sírvelos a tu navegador, scripts y agentes de programación mediante APIs compatibles con OpenAI y Anthropic.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="Panel web integrado de FLYX mostrando rendimiento en tiempo real, sesiones de contexto largo, uso de VRAM y RAM, y solicitudes recientes de agentes">
  <br><sub>Panel web integrado: gestión de modelos, chat, rendimiento en vivo, desglose de memoria y configuración de agentes en un clic.</sub>
</p>

## Características principales

- **Generación hasta 6× más rápida** — Decodificación especulativa integrada (MTP, DFlash2, DSpark y búsqueda en el prompt): **565.5 tok/s** con Qwen3.8-27B y **438.9 tok/s** con Inkling-2-Mini en una RTX 5090 ([Rendimiento](#rendimiento)).
- **Hasta 1M de contexto nativo en un PC de consumo** — Ejecuta Inkling-2-Mini con 1M de contexto en 16 GB de VRAM + 32 GB de RAM.
- **Ajustado por GPU, cero flags manuales** — Incluye tablas de ajuste de kernels medidas para 16 GPUs NVIDIA y calcula automáticamente la longitud de contexto, la precisión K/V y la distribución entre GPU y RAM según la memoria disponible. Sin adivinar capas de descarga, tamaños de lote ni tipos de caché.
- **Sin instalación, sin dependencias, ~58 MB** — Los runtimes de CUDA y C++ están enlazados estáticamente; solo se requiere el controlador NVIDIA (≥ 580). Sin CUDA Toolkit, Python ni Docker. Un único `.exe` de ~58 MB en Windows (doble clic para ejecutar) o un archivo tar.gz en Linux.
- **Panel web integrado y monitor de terminal** — Se abre directamente en una interfaz web (7 idiomas) con sugerencias guiadas para buscar y descargar modelos de Hugging Face o locales, chatear (texto e imágenes), consultar documentación de API localizada y supervisar en tiempo real el rendimiento, el uso de VRAM/RAM, sesiones, solicitudes y registros ([guía del monitor](../monitor.md)).
- **Integración en un clic con agentes de programación** — Compatible con OpenAI (Chat Completions, Responses) y Anthropic Messages con llamadas a herramientas y reutilización de prefijos en múltiples turnos. Lanza **Claude Code** o **Codex CLI** directamente en tu carpeta de proyecto desde **Integración** (Windows), o copia comandos listos para ejecutar para **DeepSeek Harness, OpenCode y Pi**.
- **Precisión verificada** — Cada kernel GPU se verifica frente a una referencia en `float64`; el formato FP8 K/V predeterminado se mantiene en **KL 0.0009–0.0016** respecto a 16-bit en 27B; la decodificación especulativa es **idéntica token a token** a la decodificación normal ([numérica](../design/numerics.md)).

## Novedades

- **`v0.2.1`** — **Velocidades verificadas en cuatro GPUs, Inkling-2-Mini con DSpark y entrada de imágenes**:
  - **Qwen3.8-27B + DFlash2** escribe código a **565.5 tok/s** en una RTX 5090 (**6.03×** la decodificación normal), **305.7** en una RTX 4090, **332.8** en una RTX 5070 Ti de 16 GB y **234.1** en una RTX 3090. El prefill W4A4 con `NVFP4` en la 5090 alcanza **15 496 tok/s**.
  - **Inkling-2-Mini + DSpark** — un nuevo modelo MoE disperso, con DSpark acoplado automáticamente: **438.9 tok/s** en código en la 5090 (**1.92×** la decodificación normal), **322.4** en la 4090, **276.2** en la 5070 Ti y **222.0** en la 3090.
  - **Visión** — entrada de imágenes desde la CLI, las APIs de OpenAI y Anthropic, y al pegar o soltar en el panel, para modelos con capacidad de visión y un proyector `mmproj`.
  - **Sigue sin dependencias** — un único `.exe` para Windows, ahora de ~57 MB.
- **`v0.2.0`** — **Primer lanzamiento open-source de FLYX**: `.exe` independiente de ~59 MB para Windows y archivo tar.gz para Linux sin dependencias, inferencia GGUF y NVFP4 de contexto completo nativo con extensión automática de K/V entre GPU y RAM, decodificación especulativa integrada (MTP / DFlash2), panel web en 7 idiomas e integración en un clic con Claude Code / Codex CLI.

## Rendimiento

Velocidad de decodificación en tokens/s, verificada en cuatro GPUs (greedy). **Normal** es la decodificación sin drafter; las dos columnas **Especulativa** activan el drafter en prompts de chat y de código, con la aceleración frente a la normal entre paréntesis.

**Qwen3.8-27B + DFlash2** — chat / mixto: 3 prompts × 128 tokens; código: 1 prompt × 128 tokens.

| GPU | Cuantización | Normal | Especulativa · chat / mixto | Especulativa · código |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | 93.8 | **315.5** (3.36×) | **565.5** (6.03×) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | 49.4 | **171.6** (3.47×) | **305.7** (6.18×) |
| **RTX 3090** (24 GB) | `UD-Q4_K_XL` | 42.8 | **131.2** (3.06×) | **234.1** (5.46×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 68.8 | **154.4** (2.24×) | **332.8** (4.84×) |

**Inkling-2-Mini + DSpark** — chat / mixto: 8 prompts × 128 tokens; código: 8 prompts × 384 tokens.

| GPU | Cuantización | Normal | Especulativa · chat / mixto | Especulativa · código |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `UD-IQ3_XXS` | 225.5 | **259.8** (1.15×) | **438.9** (1.92×) |
| **RTX 4090** (24 GB) | `UD-IQ3_XXS` | 166.7 | **202.9** (1.22×) | **322.4** (1.92×) |
| **RTX 3090** (24 GB) | `UD-IQ3_XXS` | 126.2 | **136.5** (1.08×) | **222.0** (1.75×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 150.9 | **169.3** (1.12×) | **276.2** (1.80×) |

<sub>**×** es la aceleración frente a la decodificación normal con los mismos prompts, calculada con las tasas sin redondear; **Normal** muestra la ejecución de chat / mixto. La salida especulativa coincidió token a token con la decodificación normal en todas las ejecuciones que registraron los IDs de tokens (todas salvo la ejecución mixta de Qwen en Windows). Linux, excepto la RTX 5070 Ti en Windows 11 nativo.</sub>

<sub>**También medido:** el drafter **MTP** integrado en Qwen3.8-27B (sin archivo adicional) alcanza **286.6 / 158.0 / 121.7 / 167.2 tok/s** en los prompts mixtos (5090 / 4090 / 3090 / 5070 Ti). **RTX 5090 `NVFP4`:** prefill W4A4 a **15 496 tok/s**; **203.8 tok/s** con MTP a 128K de contexto (normal: 73.4); **164.5 tok/s** a 256K con **100 %** de recuperación. Hardware, controladores y comandos exactos: [docs/benchmarks.md](../benchmarks.md#decode-and-prefill-on-four-gpus).</sub>

## Inicio rápido

### 1. Descargar y ejecutar (recomendado)

Descarga la última versión desde [GitHub Releases](https://github.com/enwove/flyx/releases) — solo necesitas un controlador NVIDIA (≥ 580):

- **Windows:** Haz doble clic en `flyx-0.2.2-windows-x86_64-cuda13.exe` (o ejecuta `.\flyx-0.2.2-windows-x86_64-cuda13.exe serve` en PowerShell). El servidor arranca y abre el panel web automáticamente (eligiendo otro puerto libre si `8080` está ocupado).
- **Linux / WSL2:** Extrae el archivo y ejecuta:

```sh
./bin/flyx serve
# O carga un GGUF de Hugging Face directamente en la caché estándar de HF:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Dos comandos más para probar: el drafter DSpark y el proyector de visión se descargan junto con el modelo:

```sh
# Inkling-2-Mini, con DSpark acoplado automáticamente
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
# Pregunta sobre una imagen
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
| Arquitecturas | Qwen3.5 / Qwen3.8 (`qwen35`), Qwen3 (`qwen3`), Inkling-2-Mini (`inkling2_loop`, MoE disperso), gpt-oss-20b / 120b (`gpt-oss`, MoE disperso, GGUF), Llama (MiniCPM5-1B) | Llama solo desde `.flyx` |
| Drafters especulativos | MTP, DFlash2, DSpark, búsqueda en el prompt | MTP y DSpark se acoplan automáticamente; DFlash2 es un archivo complementario (`--drafter-file`, o **Draft with it** en el panel); DSpark acompaña a Inkling-2-Mini |
| Visión | Entrada de imágenes con un proyector `mmproj`, descargado automáticamente del repositorio del modelo en el Hub | Hoy Qwen3.5 / Qwen3.8; Inkling-2-Mini es solo texto |
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
