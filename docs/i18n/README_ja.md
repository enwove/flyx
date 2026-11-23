<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="../brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Apache-2.0 License"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows and Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="FLYX Discord に参加"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat グループ QR"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README_zh.md">中文</a> ·
  <b>日本語</b> ·
  <a href="README_ko.md">한국어</a> ·
  <a href="README_es.md">Español</a> ·
  <a href="README_fr.md">Français</a> ·
  <a href="README_de.md">Deutsch</a>
</p>

---

<p align="center">
  <a href="#クイックスタート">クイックスタート</a> ·
  <a href="#パフォーマンス">パフォーマンス</a> ·
  <a href="../models.md">対応モデル</a> ·
  <a href="../server.md">CLI と API</a> ·
  <a href="../../llms.txt">llms.txt</a>
</p>

**FLYX** は約 50 MB の単一バイナリで動作する依存関係ゼロのローカル LLM 推論エンジンです。ネイティブな長文コンテキスト対応と GPU ごとの自動チューニングを備え、面倒なセットアップは一切不要です。GGUF モデルをそのまま読み込み、OpenAI / Anthropic 互換 API を通じてブラウザ、スクリプト、コーディングエージェントへ即座に提供できます。

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="FLYX 内蔵 Web ダッシュボード：リアルタイムスループット、長文コンテキストセッション、VRAM / RAM 使用量、最近のエージェントリクエストを表示">
  <br><sub>内蔵 Web ダッシュボード：モデル管理、チャット、リアルタイムスループット・メモリ内訳、ワンクリックのエージェント連携。</sub>
</p>

## 主な特長

- **品質劣化のないネイティブ長文コンテキスト** — コンシューマー GPU 1 枚でモデルの学習時フルコンテキストをそのまま利用可能。K/V キャッシュが VRAM を超えても切り捨てやスライディングウィンドウは行わず、システム RAM へシームレスに拡張します：**24 GB RTX 4090 の 256K で 123 tok/s**、**RTX 5090（`NVFP4`）の 256K で 165 tok/s**、さらに **32K–256K の長文検索（NIAH）で 100% 再現**（[詳細](../benchmarks.md#1-qwen38-27b-long-context-coding-benchmark)）。
- **GPU 別チューニング済み・手動フラグ不要** — 11 種類の NVIDIA GPU 向け実測カーネルチューニング表を内蔵し、利用可能なメモリに応じてコンテキスト長・K/V 精度・GPU/RAM 配置を自動決定します。オフロード層数やバッチサイズ、キャッシュ形式を試行錯誤する必要はありません。
- **インストール不要・依存関係ゼロ・約 50 MB** — CUDA および C++ ランタイムを静的リンク済み。必要なのは NVIDIA ドライバ（≥ 580）のみで、CUDA Toolkit・Python・Docker は不要です。Windows は約 50 MB の単一 `.exe`（ダブルクリックで起動）、Linux は約 45 MB のアーカイブです。
- **内蔵 Web ダッシュボード＆ターミナルモニター** — 起動すると多言語 Web UI（7 言語対応）が自動で開き、次の操作ガイドとともに Hugging Face / ローカルモデルの検索・ダウンロード、チャット、多言語 API ドキュメント閲覧、リアルタイムのスループット・VRAM/RAM 内訳・セッション・リクエスト・ログ監視が行えます（[モニターガイド](../monitor.md)）。
- **コーディングエージェントへワンクリック連携** — ツール呼び出しとマルチターン接頭辞キャッシュを備えた OpenAI（Chat Completions / Responses）および Anthropic Messages API に対応。**連携（Integration）**画面からプロジェクトフォルダを選んで **Claude Code** や **Codex CLI** を直接起動できるほか（Windows）、**DeepSeek Harness / OpenCode / Pi** 向けの即実行コマンドもコピーできます。
- **検証済みの数値精度** — すべての GPU カーネルを `float64` リファレンス実装と照合テスト済み。デフォルトの FP8 K/V は 27B モデルで 16-bit 比 **KL 0.0009–0.0016** に収まり、すべての高速化パスは**トークン単位で完全一致**の出力を保証します（[数値仕様](../design/numerics.md)）。

## 更新情報

- **`2026-11`** — **FLYX 初回オープンソースリリース**：約 50 MB の依存関係ゼロ Windows 単一 `.exe` / Linux tar.gz、GPU/RAM 自動 K/V 拡張によるネイティブ全コンテキスト GGUF & NVFP4 推論、内蔵投機デコーディング（MTP / DFlash2）、7 言語対応 Web ダッシュボード、ワンクリック Claude Code / Codex CLI 連携を提供。

## パフォーマンス

**Qwen3.8-27B GGUF** で内蔵 **MTP** 投機デコーディング（`--drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos`）を使用し、**1K / 128K / 256K** コンテキストにおける 512 トークン貪欲コード生成を測定しました（RTX 5090 は `NVFP4`、24 GB GPU は `UD-Q4_K_XL`、16 GB GPU は `UD-IQ3_XXS`。正確なプリフィル時間、262K 自動メモリ配置、`flyx bench` ドラフター比較表、再現コマンドは [docs/benchmarks.md](../benchmarks.md) を参照）：

| GPU | モデル | 1K 生成 | 128K 生成 | 256K 生成 | 256K 検索（NIAH） |
|---|---|---:|---:|---:|:---:|
| **RTX 5090** (32 GB) | `NVFP4` | **350.2 tok/s** | **228.7 tok/s** | **164.5 tok/s** | **100%** (4/4 マルチニードル) |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` | **225.9 tok/s** | **161.0 tok/s** | **105.8 tok/s** | **100%** (5/5 深度グリッド) |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` | **167.4 tok/s** | **104.6 tok/s** | **61.2 tok/s** | **100%** (5/5 + 4/4 マルチ) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | **193.6 tok/s** | **155.9 tok/s** | **122.7 tok/s** | **100%** (20/20 + 4/4 マルチ) |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` | **172.3 tok/s** | **111.0 tok/s** | **99.0 tok/s** | **100%** (3/3 @ 1K–256K) |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` | **135.8 tok/s** | **97.5 tok/s** | **68.1 tok/s** | **100%** (3/3 @ 1K–256K) |

<sub>**RTX 5090 + DFlash2（`NVFP4`, `--drafter all`）：** 1K で **365.0 tok/s**、128K で **273.1 tok/s**（プリフィル `30.1 s`）。`flyx bench` 混合プロンプトは **311.6 tok/s**（`DFlash2`）。**長文コンテキスト検索（Needle-in-a-Haystack）：** 全 6 GPU で **51/51（100%）**完全一致（32K–256K の `5%–95%` 深度グリッドおよび 256K 4 ニードル検索、フル GPU FP8 と Host-RAM + 2-bit StrataFlux の双方を網羅、[全グリッド](../benchmarks.md#3-long-context-attention-fidelity-needle-in-a-haystack-suite)）。</sub>

## クイックスタート

### 1. ダウンロードして実行（推奨）

[GitHub Releases](https://github.com/enwove/flyx/releases) から最新版をダウンロードしてください。必要なのは NVIDIA ドライバ（≥ 580）のみです：

- **Windows：** `flyx-*-windows-x86_64-cuda13.exe` をダブルクリック（または PowerShell で `.\flyx.exe serve` を実行）。サーバーが起動し、自動でダッシュボードが開きます（`8080` が使用中の場合は別の空きポートを自動選択）。
- **Linux / WSL2：** アーカイブを展開して実行：

```sh
./bin/flyx serve
# または Hugging Face の GGUF を指定して標準 HF キャッシュへ直接読み込み：
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

ブラウザで `http://127.0.0.1:8080` を開き、**モデル（Model）**画面でモデルを選択またはダウンロードした後、**連携（Integration）**画面から **Claude Code / Codex** を直接起動（または各エージェント用コマンドをコピー）できます。

| クライアント / エンドポイント | ベース URL |
|---|---|
| OpenAI SDK（Chat Completions, Responses, Completions） | `http://127.0.0.1:8080/v1` |
| Anthropic SDK・Claude Code（`ANTHROPIC_BASE_URL`） | `http://127.0.0.1:8080` |
| Web ダッシュボード | `http://127.0.0.1:8080/` |
| テレメトリ JSON（`flyx monitor`） | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>ソースからビルド</b>（CUDA 13, CMake 3.24+, Ninja, g++ 11+ / MSVC 2022）</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # 出力先: ~/flyx-build/bin

# Hugging Face 名を指定して直接実行またはサーバー起動
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "GPUについての俳句を詠んでください。"
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **WSL2 での注意：** ビルドディレクトリとモデルファイルは Linux ファイルシステム（`~/models`, `~/flyx-build`）に配置してください。`/mnt/<drive>` 上ではモデル読み込み速度が大幅に低下します。

</details>

<details>
<summary><b>Python SDK と curl の使用例</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "こんにちは！"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "フランスの首都を教えてください。"}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## 対応モデル・ハードウェア

| 項目 | 対応範囲 | 備考 |
|---|---|---|
| アーキテクチャ | Qwen3.5 / Qwen3.8（ハイブリッド GDN + ゲート付きアテンション）、Qwen3 Dense、Llama（MiniCPM5-1B） | テキスト専用 Dense モデルのみ（MoE・ビジョン非対応）。Llama は `.flyx` 形式のみ |
| モデル形式 | GGUF（`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, GGML NVFP4）；`.flyx`（BF16, NVFP4；フォーマット 2 では対象モデル付属の LoRA アダプター V-LoRA を未マージのまま実行可能。DV ドラフターは `--drafter dv` 指定時のみ有効、[設計](../design/dv.md)） | MXFP4 は非対応。分割 GGUF はすべてのパートを同一フォルダに配置 |
| GPU | RTX 30 / 40 / 50 シリーズ、L4、L40S、A100、H100（`sm_80`–`sm_120`、Linux リリースは `sm_100` も収録） | 1 プロセス 1 GPU（`CUDA_VISIBLE_DEVICES`）、ドライバ ≥ 580 |
| プラットフォーム | Windows（ネイティブ MSVC）、Linux、WSL2 | Windows は単一 `.exe` / ZIP / wheel、Linux は tar.gz / `manylinux` wheel |

## ドキュメント

AI エージェントは [llms.txt](../../llms.txt) から開始してください。

- **利用・連携：** [docs/server.md](../server.md) · [docs/models.md](../models.md) · [docs/monitor.md](../monitor.md) · [docs/benchmarks.md](../benchmarks.md)
- **ビルド・テスト：** [docs/building.md](../building.md) · [docs/testing.md](../testing.md)
- **設計・数値仕様：** [docs/design/architecture.md](../design/architecture.md) · [docs/design/numerics.md](../design/numerics.md) · [docs/design/gguf.md](../design/gguf.md) · [docs/design/tree-verification.md](../design/tree-verification.md) · [docs/design/dv.md](../design/dv.md)
- **コントリビュート・ガバナンス：** [AGENTS.md](../../AGENTS.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [SECURITY.md](../../SECURITY.md) · [docs/prior-art.md](../prior-art.md)

## お問い合わせ

- **Discord：** [FLYX Discord に参加](https://discord.gg/ZtUPXBxYhN)
- **WeChat：** [WeChat グループの QR コードを表示](https://enwove.ai/flyx/wechatqr)
- **不具合報告・ご要望：** [GitHub Issues](https://github.com/enwove/flyx/issues)

協業やパートナーシップに関するお問い合わせは [contact@enwove.ai](mailto:contact@enwove.ai) までご連絡ください。

## ライセンス

Apache-2.0。サードパーティ帰属表示は [NOTICE](../../NOTICE) および [docs/prior-art.md](../prior-art.md) に記載しています。
