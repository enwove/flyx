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

**FLYX** は依存関係ゼロのローカル LLM 推論エンジンで、Windows では約 57 MB の単一 `.exe` に収まります。高速な投機デコード、ネイティブな長文コンテキスト、GPU ごとの自動チューニングを備え、面倒なセットアップは一切不要です。対応する GGUF モデルをそのまま読み込み、OpenAI / Anthropic 互換 API を通じてブラウザ、スクリプト、コーディングエージェントへ即座に提供できます。

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="FLYX 内蔵 Web ダッシュボード：リアルタイムスループット、長文コンテキストセッション、VRAM / RAM 使用量、最近のエージェントリクエストを表示">
  <br><sub>内蔵 Web ダッシュボード：モデル管理、チャット、リアルタイムスループット・メモリ内訳、ワンクリックのエージェント連携。</sub>
</p>

## 主な特長

- **生成を最大 6× 高速化** — 投機デコード（MTP・DFlash2・DSpark・プロンプト検索）を内蔵。RTX 5090 で Qwen3.8-27B は **565.5 tok/s**、Inkling-2-Mini は **438.9 tok/s** に達します（[パフォーマンス](#パフォーマンス)）。
- **コンシューマー PC で最大 1M のネイティブコンテキスト** — 16 GB VRAM + 32 GB RAM で Inkling-2-Mini を 1M コンテキストで実行できます。
- **GPU 別チューニング済み・手動フラグ不要** — 16 種類の NVIDIA GPU 向け実測カーネルチューニング表を内蔵し、利用可能なメモリに応じてコンテキスト長・K/V 精度・GPU/RAM 配置を自動決定します。オフロード層数やバッチサイズ、キャッシュ形式を試行錯誤する必要はありません。
- **インストール不要・依存関係ゼロ・約 57 MB** — CUDA および C++ ランタイムを静的リンク済み。必要なのは NVIDIA ドライバ（≥ 580）のみで、CUDA Toolkit・Python・Docker は不要です。Windows は約 57 MB の単一 `.exe`（ダブルクリックで起動）、Linux は tar.gz アーカイブです。
- **内蔵 Web ダッシュボード＆ターミナルモニター** — 起動すると多言語 Web UI（7 言語対応）が自動で開き、次の操作ガイドとともに Hugging Face / ローカルモデルの検索・ダウンロード、チャット（テキストと画像）、多言語 API ドキュメント閲覧、リアルタイムのスループット・VRAM/RAM 内訳・セッション・リクエスト・ログ監視が行えます（[モニターガイド](../monitor.md)）。
- **コーディングエージェントへワンクリック連携** — ツール呼び出しとマルチターン接頭辞キャッシュを備えた OpenAI（Chat Completions / Responses）および Anthropic Messages API に対応。**連携（Integration）**画面からプロジェクトフォルダを選んで **Claude Code** や **Codex CLI** を直接起動できるほか（Windows）、**DeepSeek Harness / OpenCode / Pi** 向けの即実行コマンドもコピーできます。
- **検証済みの数値精度** — すべての GPU カーネルを `float64` リファレンス実装と照合テスト済み。デフォルトの FP8 K/V は 27B モデルで 16-bit 比 **KL 0.0009–0.0016** に収まり、投機デコードの出力は通常デコードと**トークン単位で完全一致**します（[数値仕様](../design/numerics.md)）。

## 更新情報

- **`v0.2.1`** — **4 種の GPU で検証した速度、Inkling-2-Mini と DSpark、画像入力**：
  - **Qwen3.8-27B + DFlash2** のコード生成は RTX 5090 で **565.5 tok/s**（通常デコードの **6.03×**）、RTX 4090 で **305.7**、16 GB の RTX 5070 Ti で **332.8**、RTX 3090 で **234.1**。5090 での `NVFP4` W4A4 プリフィルは **15,496 tok/s** です。
  - **Inkling-2-Mini + DSpark** — 新たに対応した疎な MoE モデルで、DSpark は自動で接続されます。コード生成は 5090 で **438.9 tok/s**（通常デコードの **1.92×**）、4090 で **322.4**、5070 Ti で **276.2**、3090 で **222.0**。
  - **ビジョン** — `mmproj` プロジェクターを備えたビジョン対応モデルで、CLI、OpenAI / Anthropic API、ダッシュボードへの貼り付け / ドロップから画像を入力できます。
  - **引き続き依存関係ゼロ** — Windows は単一の `.exe`、現在は約 57 MB。
- **`v0.2.0`** — **FLYX 初回オープンソースリリース**：約 59 MB の依存関係ゼロ Windows 単一 `.exe` / Linux tar.gz、GPU/RAM 自動 K/V 拡張によるネイティブ全コンテキスト GGUF & NVFP4 推論、内蔵投機デコーディング（MTP / DFlash2）、7 言語対応 Web ダッシュボード、ワンクリック Claude Code / Codex CLI 連携を提供。

## パフォーマンス

4 種の GPU で検証した v0.2.1 のデコード速度（トークン/秒、greedy デコード）。**通常**はドラフターなしのデコード、2 つの**投機**列はチャットとコードのプロンプトでドラフターを有効にした結果で、括弧内は通常デコードに対する高速化率です。

**Qwen3.8-27B + DFlash2** — チャット / 混合：プロンプト 3 件 × 128 トークン、コード：プロンプト 1 件 × 128 トークン。

| GPU | 量子化 | 通常 | 投機 · チャット / 混合 | 投機 · コード |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | 93.8 | **315.5** (3.36×) | **565.5** (6.03×) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | 49.4 | **171.6** (3.47×) | **305.7** (6.18×) |
| **RTX 3090** (24 GB) | `UD-Q4_K_XL` | 42.8 | **131.2** (3.06×) | **234.1** (5.46×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 68.8 | **154.4** (2.24×) | **332.8** (4.84×) |

**Inkling-2-Mini + DSpark** — チャット / 混合：プロンプト 8 件 × 128 トークン、コード：プロンプト 8 件 × 384 トークン。

| GPU | 量子化 | 通常 | 投機 · チャット / 混合 | 投機 · コード |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `UD-IQ3_XXS` | 225.5 | **259.8** (1.15×) | **438.9** (1.92×) |
| **RTX 4090** (24 GB) | `UD-IQ3_XXS` | 166.7 | **202.9** (1.22×) | **322.4** (1.92×) |
| **RTX 3090** (24 GB) | `UD-IQ3_XXS` | 126.2 | **136.5** (1.08×) | **222.0** (1.75×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 150.9 | **169.3** (1.12×) | **276.2** (1.80×) |

<sub>**×** は同じプロンプトでの通常デコードに対する高速化率で、丸める前の速度から算出しています。**通常**列はチャット / 混合の測定値です。トークン ID を記録したすべての測定で、投機デコードの出力は通常デコードとトークン単位で一致しました（ID を記録していないのは Windows の Qwen 混合測定のみ）。RTX 5070 Ti のみネイティブ Windows 11、ほかは Linux です。</sub>

<sub>**そのほかの測定：** Qwen3.8-27B 内蔵の **MTP** ドラフター（追加ファイル不要）は、混合プロンプトで **286.6 / 158.0 / 121.7 / 167.2 tok/s**（5090 / 4090 / 3090 / 5070 Ti）。**RTX 5090 `NVFP4`：** W4A4 プリフィルは **15,496 tok/s**、128K コンテキストでの MTP は **203.8 tok/s**（通常：73.4）、256K では **164.5 tok/s** で長文検索の再現率 **100%**（v0.2.0 の測定）。ハードウェア、ドライバ、実行コマンドは [docs/benchmarks.md](../benchmarks.md#v021-four-gpu-verified-benchmarks) を参照してください。</sub>

## クイックスタート

### 1. ダウンロードして実行（推奨）

[GitHub Releases](https://github.com/enwove/flyx/releases) から最新版をダウンロードしてください。必要なのは NVIDIA ドライバ（≥ 580）のみです：

- **Windows：** `flyx-0.2.1-windows-x86_64-cuda13.exe` をダブルクリック（または PowerShell で `.\flyx-0.2.1-windows-x86_64-cuda13.exe serve` を実行）。サーバーが起動し、自動でダッシュボードが開きます（`8080` が使用中の場合は別の空きポートを自動選択）。
- **Linux / WSL2：** アーカイブを展開して実行：

```sh
./bin/flyx serve
# または Hugging Face の GGUF を指定して標準 HF キャッシュへ直接読み込み：
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

あと 2 つお試しください。DSpark ドラフターとビジョン用プロジェクターはモデルと一緒にダウンロードされます：

```sh
# Inkling-2-Mini（DSpark を自動で接続）
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
# 画像について質問する
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
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
| アーキテクチャ | Qwen3.5 / Qwen3.8（`qwen35`）、Qwen3（`qwen3`）、Inkling-2-Mini（`inkling2_loop`、疎な MoE）、Llama（MiniCPM5-1B） | Llama は `.flyx` のみ |
| 投機ドラフター | MTP、DFlash2、DSpark、プロンプト検索 | MTP と DSpark は自動で接続。DFlash2 は付属ファイル（`--drafter-file`、またはダッシュボードの **Draft with it**）。DSpark は Inkling-2-Mini 用 |
| ビジョン | `mmproj` プロジェクターによる画像入力（モデルの Hub リポジトリから自動取得） | 現在は Qwen3.5 / Qwen3.8 に対応。Inkling-2-Mini はテキストのみ |
| モデル形式 | GGUF（`F32`, `F16`, `BF16`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, `MXFP4`, `Q2_0`, GGML NVFP4）；`.flyx`（BF16, NVFP4；フォーマット 2 では対象モデル付属の LoRA アダプター V-LoRA を未マージのまま実行可能。DV ドラフターは `--drafter dv` 指定時のみ有効、[設計](../design/dv.md)） | 分割 GGUF の全ファイルは同じフォルダーに置きます。 |
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
