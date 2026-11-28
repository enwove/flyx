<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="../brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Apache-2.0 License"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows and Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="FLYX Discord 참여"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat 그룹 QR"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README_zh.md">中文</a> ·
  <a href="README_ja.md">日本語</a> ·
  <b>한국어</b> ·
  <a href="README_es.md">Español</a> ·
  <a href="README_fr.md">Français</a> ·
  <a href="README_de.md">Deutsch</a>
</p>

---

<p align="center">
  <a href="#빠른-시작">빠른 시작</a> ·
  <a href="#성능">성능</a> ·
  <a href="../models.md">지원 모델</a> ·
  <a href="../server.md">CLI 및 API</a> ·
  <a href="../../llms.txt">llms.txt</a>
</p>

**FLYX**는 의존성 제로 로컬 LLM 추론 엔진으로, Windows에서는 약 58 MB 단일 `.exe` 하나에 담깁니다. 빠른 추측 디코딩, 네이티브 롱 컨텍스트, GPU별 자동 튜닝을 제공하며 별도의 설정이 필요하지 않습니다. 지원하는 GGUF 모델을 변환 없이 그대로 실행하고 OpenAI 및 Anthropic 호환 API를 통해 브라우저, 스크립트, 코딩 에이전트에 바로 연결할 수 있습니다.

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="실시간 처리량, 롱 컨텍스트 세션, VRAM/RAM 사용량 및 최근 에이전트 요청을 표시하는 FLYX 내장 웹 대시보드">
  <br><sub>내장 웹 대시보드: 모델 관리, 채팅, 실시간 처리량 및 메모리 분석, 원클릭 코딩 에이전트 연동.</sub>
</p>

## 주요 기능

- **최대 6× 빠른 생성** — 추측 디코딩(MTP, DFlash2, DSpark, 프롬프트 룩업)을 내장했습니다. RTX 5090에서 Qwen3.8-27B는 **565.5 tok/s**, Inkling-2-Mini는 **438.9 tok/s**에 이릅니다([성능](#성능)).
- **소비자용 PC에서 최대 1M 네이티브 컨텍스트** — 16 GB VRAM + 32 GB RAM에서 Inkling-2-Mini를 1M 컨텍스트로 실행합니다.
- **GPU별 사전 튜닝, 수동 플래그 불필요** — 16종의 NVIDIA GPU에 대한 실측 커널 튜닝 테이블을 내장하고 있으며, 가용 메모리를 기준으로 컨텍스트 길이, K/V 정밀도, GPU/RAM 배치를 자동 결정합니다. 오프로드 레이어 수, 배치 크기, 캐시 타입을 직접 맞출 필요가 없습니다.
- **설치 불필요, 의존성 제로, 약 58 MB** — CUDA 및 C++ 런타임이 정적으로 링크되어 있어 NVIDIA 드라이버(≥ 580)만 있으면 됩니다. CUDA Toolkit, Python, Docker가 필요 없습니다. Windows에서는 약 58 MB 단일 `.exe`(더블 클릭 실행), Linux에서는 tar.gz 압축 파일로 제공됩니다.
- **내장 웹 대시보드 및 터미널 모니터** — 실행 즉시 7개 언어를 지원하는 브라우저 UI가 열리며, 단계별 안내와 함께 Hugging Face 또는 로컬 모델 검색·다운로드, 채팅(텍스트 및 이미지), 현지화된 API 문서 확인, 실시간 처리량·VRAM/RAM 분할·세션·요청·로그 모니터링을 제공합니다([모니터 가이드](../monitor.md)).
- **원클릭 코딩 에이전트 연동** — 도구 호출과 멀티턴 프롬프트 재사용을 지원하는 OpenAI(Chat Completions, Responses) 및 Anthropic Messages API를 제공합니다. **연동(Integration)** 탭에서 프로젝트 폴더를 선택해 **Claude Code** 또는 **Codex CLI**를 바로 실행할 수 있으며(Windows), **DeepSeek Harness, OpenCode, Pi**용 실행 명령도 원클릭 복사할 수 있습니다.
- **검증된 정확도** — 모든 GPU 커널을 `float64` 레퍼런스 구현과 대조 테스트합니다. 기본 FP8 K/V는 27B 모델에서 16-bit 대비 **KL 0.0009–0.0016** 이내를 유지하며, 추측 디코딩의 출력은 기본 디코딩과 **토큰 단위로 완전히 동일**합니다([수치 규격](../design/numerics.md)).

## 업데이트

- **`v0.2.1`** — **GPU 4종에서 검증한 속도, Inkling-2-Mini와 DSpark, 이미지 입력**:
  - **Qwen3.8-27B + DFlash2**의 코드 생성 속도는 RTX 5090에서 **565.5 tok/s**(기본 디코딩의 **6.03×**), RTX 4090에서 **305.7**, 16 GB RTX 5070 Ti에서 **332.8**, RTX 3090에서 **234.1**입니다. 5090의 `NVFP4` W4A4 프리필은 **15,496 tok/s**입니다.
  - **Inkling-2-Mini + DSpark** — 새로 지원하는 희소 MoE 모델이며 DSpark가 자동으로 연결됩니다. 코드 생성은 5090에서 **438.9 tok/s**(기본 디코딩의 **1.92×**), 4090에서 **322.4**, 5070 Ti에서 **276.2**, 3090에서 **222.0**입니다.
  - **비전** — `mmproj` 프로젝터를 갖춘 비전 지원 모델에 CLI, OpenAI 및 Anthropic API, 대시보드 붙여넣기 / 드롭으로 이미지를 입력할 수 있습니다.
  - **여전히 의존성 제로** — Windows 단일 `.exe`, 이제 약 57 MB.
- **`v0.2.0`** — **FLYX 첫 오픈소스 릴리스**: 약 59 MB 무의존성 Windows 단일 `.exe` 및 Linux tar.gz, 자동 GPU/RAM K/V 확장을 갖춘 네이티브 풀 컨텍스트 GGUF & NVFP4 추론, 내장 추측 디코딩(MTP / DFlash2), 7개 언어 웹 대시보드, 원클릭 Claude Code / Codex CLI 연동 지원.

## 성능

GPU 4종에서 검증한 디코딩 속도(토큰/초, greedy)입니다. **기본**은 드래프터 없이 디코딩한 값이고, 두 **추측** 열은 채팅과 코드 프롬프트에서 드래프터를 켠 값이며 괄호 안은 기본 디코딩 대비 속도 향상 배수입니다.

**Qwen3.8-27B + DFlash2** — 채팅 / 혼합: 프롬프트 3개 × 128토큰, 코드: 프롬프트 1개 × 128토큰.

| GPU | 양자화 | 기본 | 추측 · 채팅 / 혼합 | 추측 · 코드 |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | 93.8 | **315.5** (3.36×) | **565.5** (6.03×) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | 49.4 | **171.6** (3.47×) | **305.7** (6.18×) |
| **RTX 3090** (24 GB) | `UD-Q4_K_XL` | 42.8 | **131.2** (3.06×) | **234.1** (5.46×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 68.8 | **154.4** (2.24×) | **332.8** (4.84×) |

**Inkling-2-Mini + DSpark** — 채팅 / 혼합: 프롬프트 8개 × 128토큰, 코드: 프롬프트 8개 × 384토큰.

| GPU | 양자화 | 기본 | 추측 · 채팅 / 혼합 | 추측 · 코드 |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `UD-IQ3_XXS` | 225.5 | **259.8** (1.15×) | **438.9** (1.92×) |
| **RTX 4090** (24 GB) | `UD-IQ3_XXS` | 166.7 | **202.9** (1.22×) | **322.4** (1.92×) |
| **RTX 3090** (24 GB) | `UD-IQ3_XXS` | 126.2 | **136.5** (1.08×) | **222.0** (1.75×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 150.9 | **169.3** (1.12×) | **276.2** (1.80×) |

<sub>**×**는 같은 프롬프트에서 기본 디코딩 대비 속도 향상 배수이며 반올림 전 속도로 계산했습니다. **기본** 열은 채팅 / 혼합 측정값입니다. 토큰 ID를 기록한 모든 측정에서 추측 디코딩의 출력은 기본 디코딩과 토큰 단위로 일치했습니다(ID를 기록하지 않은 것은 Windows의 Qwen 혼합 측정뿐입니다). RTX 5070 Ti만 네이티브 Windows 11이고 나머지는 Linux입니다.</sub>

<sub>**추가 측정:** Qwen3.8-27B에 내장된 **MTP** 드래프터(추가 파일 불필요)는 혼합 프롬프트에서 **286.6 / 158.0 / 121.7 / 167.2 tok/s**입니다(5090 / 4090 / 3090 / 5070 Ti). **RTX 5090 `NVFP4`:** W4A4 프리필 **15,496 tok/s**, 128K 컨텍스트에서 MTP **203.8 tok/s**(기본: 73.4), 256K에서 **164.5 tok/s**와 롱 컨텍스트 검색 정확도 **100%**(v0.2.0 측정). 하드웨어, 드라이버, 정확한 명령어는 [docs/benchmarks.md](../benchmarks.md#decode-and-prefill-on-four-gpus)를 참고하세요.</sub>

## 빠른 시작

### 1. 다운로드 후 바로 실행 (권장)

[GitHub Releases](https://github.com/enwove/flyx/releases)에서 최신 버전을 다운로드하세요. NVIDIA 드라이버(≥ 580)만 있으면 됩니다:

- **Windows:** `flyx-0.2.2-windows-x86_64-cuda13.exe`를 더블 클릭(또는 PowerShell에서 `.\flyx-0.2.2-windows-x86_64-cuda13.exe serve` 실행)하면 서버가 시작되고 대시보드가 자동으로 열립니다(`8080` 포트가 사용 중이면 다른 빈 포트를 자동 선택).
- **Linux / WSL2:** 압축을 풀고 다음을 실행합니다:

```sh
./bin/flyx serve
# 또는 Hugging Face GGUF를 표준 HF 캐시로 직접 불러와 실행:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

두 가지를 더 실행해 보세요. DSpark 드래프터와 비전 프로젝터는 모델과 함께 다운로드됩니다:

```sh
# Inkling-2-Mini (DSpark 자동 연결)
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
# 이미지에 대해 질문하기
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
```

브라우저에서 `http://127.0.0.1:8080`을 열고 **모델(Model)** 탭에서 모델을 선택하거나 다운로드한 뒤, **연동(Integration)** 탭에서 **Claude Code / Codex**를 바로 실행하거나 다른 에이전트 명령을 복사하세요.

| 클라이언트 / 엔드포인트 | 기본 URL |
|---|---|
| OpenAI SDK (Chat Completions, Responses, Completions) | `http://127.0.0.1:8080/v1` |
| Anthropic SDK 및 Claude Code (`ANTHROPIC_BASE_URL`) | `http://127.0.0.1:8080` |
| 웹 대시보드 | `http://127.0.0.1:8080/` |
| 텔레메트리 JSON (`flyx monitor`) | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>소스에서 빌드</b> (CUDA 13, CMake 3.24+, Ninja, g++ 11+ / MSVC 2022)</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # 출력 위치: ~/flyx-build/bin

# Hugging Face 이름으로 직접 실행 또는 서빙
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "GPU에 대한 짧은 시를 써 주세요."
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **WSL2 참고:** 빌드 디렉토리와 모델 파일은 Linux 파일시스템(`~/models`, `~/flyx-build`)에 두세요. `/mnt/<drive>` 경로에서는 모델 로딩 속도가 크게 느려집니다.

</details>

<details>
<summary><b>Python SDK 및 curl 예제</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "안녕하세요!"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "프랑스의 수도는 어디인가요?"}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## 지원 모델 및 하드웨어

| 항목 | 지원 범위 | 참고 |
|---|---|---|
| 아키텍처 | Qwen3.5 / Qwen3.8 (`qwen35`), Qwen3 (`qwen3`), Inkling-2-Mini (`inkling2_loop`, 희소 MoE), gpt-oss-20b / 120b (`gpt-oss`, 희소 MoE, GGUF), Llama (MiniCPM5-1B) | Llama는 `.flyx`만 지원 |
| 추측 드래프터 | MTP, DFlash2, DSpark, 프롬프트 룩업 | MTP와 DSpark는 자동으로 연결됩니다. DFlash2는 별도 파일입니다(`--drafter-file` 또는 대시보드의 **Draft with it**). DSpark는 Inkling-2-Mini 전용입니다 |
| 비전 | `mmproj` 프로젝터를 통한 이미지 입력(모델의 Hub 저장소에서 자동으로 가져옴) | 현재 Qwen3.5 / Qwen3.8 지원. Inkling-2-Mini는 텍스트 전용 |
| 모델 형식 | GGUF (`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, `MXFP4`, `Q2_0`, GGML NVFP4); `.flyx` (BF16, NVFP4; 포맷 2는 타깃 모델의 LoRA 어댑터인 V-LoRA를 병합 없이 실행 가능하며, DV 드래프터는 `--drafter dv` 지정 시에만 사용, [설계](../design/dv.md)) | 분할 GGUF 파일은 모두 같은 폴더에 둡니다. |
| GPU | RTX 30 / 40 / 50 시리즈, L4, L40S, A100, H100 (`sm_80`–`sm_120`; Linux 릴리스는 `sm_100` 포함) | 프로세스당 단일 GPU (`CUDA_VISIBLE_DEVICES`); 드라이버 ≥ 580 |
| 플랫폼 | Windows (네이티브 MSVC), Linux, WSL2 | Windows는 단일 `.exe` / ZIP / wheel; Linux는 tar.gz / `manylinux` wheel |

## 문서

AI 에이전트는 [llms.txt](../../llms.txt)부터 확인하세요.

- **사용 및 연동:** [docs/server.md](../server.md) · [docs/models.md](../models.md) · [docs/monitor.md](../monitor.md) · [docs/benchmarks.md](../benchmarks.md)
- **빌드 및 테스트:** [docs/building.md](../building.md) · [docs/testing.md](../testing.md)
- **설계 및 규격:** [docs/design/architecture.md](../design/architecture.md) · [docs/design/numerics.md](../design/numerics.md) · [docs/design/gguf.md](../design/gguf.md) · [docs/design/tree-verification.md](../design/tree-verification.md) · [docs/design/dv.md](../design/dv.md)
- **기여 및 거버넌스:** [AGENTS.md](../../AGENTS.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [SECURITY.md](../../SECURITY.md) · [docs/prior-art.md](../prior-art.md)

## 문의하기

- **Discord:** [FLYX Discord 참여](https://discord.gg/ZtUPXBxYhN)
- **WeChat:** [WeChat 그룹 QR 코드 보기](https://enwove.ai/flyx/wechatqr)
- **이슈 및 피드백:** [GitHub Issues](https://github.com/enwove/flyx/issues)

협업 및 파트너십 문의는 [contact@enwove.ai](mailto:contact@enwove.ai)로 연락해 주세요.

## 라이선스

Apache-2.0. 서드파티 고지는 [NOTICE](../../NOTICE)와 [docs/prior-art.md](../prior-art.md)에 명시되어 있습니다.
