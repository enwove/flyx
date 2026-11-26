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

**FLYX** は依存関係ゼロのローカル LLM 推論エンジンです。対応する GGUF モデルを直接読み込み、OpenAI・Anthropic 互換 API でブラウザやスクリプト、コーディングエージェントから利用できます。

<p align="center">
  <img src="../media/dashboard-light.png" width="900" alt="FLYX 内蔵 Web ダッシュボード：リアルタイムスループット、長文コンテキストセッション、VRAM / RAM 使用量、最近のエージェントリクエストを表示">
  <br><sub>内蔵 Web ダッシュボード：モデル管理、チャット、リアルタイムスループット・メモリ内訳、ワンクリックのエージェント連携。</sub>
</p>

## 主な特長

- 長いコンテキストには VRAM とシステム RAM を使用します。片方のメモリ予算を増やせば指定長に届く場合、必要な追加バイト数を表示します。
- **GPU 別チューニング済み・手動フラグ不要** — 11 種類の NVIDIA GPU 向け実測カーネルチューニング表を内蔵し、利用可能なメモリに応じてコンテキスト長・K/V 精度・GPU/RAM 配置を自動決定します。オフロード層数やバッチサイズ、キャッシュ形式を試行錯誤する必要はありません。
- **インストール不要・依存関係ゼロ・約 57 MB** — Windows の単一実行ファイルには CUDA と C++ のランタイムが含まれます。
- **内蔵 Web ダッシュボード＆ターミナルモニター** — 起動すると多言語 Web UI（7 言語対応）が自動で開き、次の操作ガイドとともに Hugging Face / ローカルモデルの検索・ダウンロード、チャット、多言語 API ドキュメント閲覧、リアルタイムのスループット・VRAM/RAM 内訳・セッション・リクエスト・ログ監視が行えます（[モニターガイド](../monitor.md)）。
- **コーディングエージェントへワンクリック連携** — ツール呼び出しとマルチターン接頭辞キャッシュを備えた OpenAI（Chat Completions / Responses）および Anthropic Messages API に対応。**連携（Integration）**画面からプロジェクトフォルダを選んで **Claude Code** や **Codex CLI** を直接起動できるほか（Windows）、**DeepSeek Harness / OpenCode / Pi** 向けの即実行コマンドもコピーできます。
- CUDA カーネルは float64 の参照実装と比較しています。ベンチマークには投機実行と通常デコードの出力 ID が一致した GPU とプロンプトを記載しています。 [float64](../design/numerics.md).

## 更新情報

- **v0.2.1** — Inkling-2-Mini の疎な MoE とデコードカーネルを最適化し、Markov ドラフター DSpark を自動で読み込みます。Qwen3.5/3.8 は対応プロジェクターで画像入力が可能です。Hub が付属ファイルを検出し、**Draft with it** でドラフターを選べます。空き VRAM に応じてドラフターの読み込みを調整し、DFlash2 のスライディングウィンドウにはリング KV キャッシュを使います。PTX を含まない Windows 候補版は 57.14 MB で、v0.2.0 の 58.96 MB より小さくなりました。
- **`v0.2.0`** — **FLYX 初回オープンソースリリース**：約 59 MB の依存関係ゼロ Windows 単一 `.exe` / Linux tar.gz、GPU/RAM 自動 K/V 拡張によるネイティブ全コンテキスト GGUF & NVFP4 推論、内蔵投機デコーディング（MTP / DFlash2）、7 言語対応 Web ダッシュボード、ワンクリック Claude Code / Codex CLI 連携を提供。

## パフォーマンス

Qwen3.8-27B のデコード速度（トークン/秒）。最初の3列は混合プロンプト3件で各128トークンを生成した結果です。最後の列は最初のコードプロンプト1件で128トークンを生成し、プロンプト検索を無効にしています。

| GPU | Qwen3.8-27B | 通常 | MTP | DFlash2 | コード DFlash2 |
|---|---|---:|---:|---:|---:|
| RTX 5090 · Linux | NVFP4 | 93.8 | 286.6 | 315.5 | 565.5 |
| RTX 4090 · Linux | UD-Q4_K_XL | 49.4 | 158.0 | 171.6 | 305.7 |
| RTX 3090 · Linux | UD-Q4_K_XL | 42.8 | 121.7 | 131.2 | 234.1 |
| RTX 5070 Ti · Windows | UD-IQ3_XXS | 68.8 | 167.2 | 154.4 | 332.8 |

Inkling-2-Mini（UD-IQ3_XXS）の通常デコード → Markov DSpark（Q8_0）の速度（トークン/秒）。各比較で出力トークン ID が一致しました。

| GPU | chat_short · 8 × 128 | code_short · 8 × 384 |
|---|---:|---:|
| RTX 5090 · Linux | 225.5 → 259.8 | 228.5 → 438.9 |
| RTX 4090 · Linux | 166.7 → 202.9 | 167.9 → 322.4 |
| RTX 3090 · Linux | 126.2 → 136.5 | 126.9 → 222.0 |
| RTX 5070 Ti · Windows | 150.9 → 169.3 | 153.6 → 276.2 |

[測定条件とコマンド](../benchmarks.md#v021-four-gpu-verified-benchmarks)

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

Linux での Hub ダウンロードにはシステムの libcurl を使います。

最初のコマンドは DSpark を自動で読み込みます。次のコマンドは Hub のプロジェクターを使って Qwen で画像を処理します。

```sh
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
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
| アーキテクチャ | GGUF: Qwen3 (`qwen3`), Qwen3.5/3.8 (`qwen35`), Inkling-2-Mini (`inkling2_loop`) | Qwen3.5/3.8 + `qwen3vl_merger` · Inkling-2-Mini + Markov DSpark · DFlash2 / DSpark: `dflash` · Llama: `.flyx` |
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
