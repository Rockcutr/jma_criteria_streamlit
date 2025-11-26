import json
import io
import os
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup


import streamlit as st
from pypdf import PdfReader
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =========================
#  OpenAI クライアント初期化
# =========================

# .env などから環境変数を読み込む
load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY が設定されていません。StreamlitのSecretsか.envを確認してください。")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================
#  PDF テキスト抽出
# =========================

def extract_text_from_pdf(uploaded_file) -> str:
    """
    アップロードされた PDF ファイルからテキストを抽出して 1 つの文字列にまとめる。
    β版ではシンプルに「全ページ結合＋上限カット」。
    """
    reader = PdfReader(uploaded_file)
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            # うまく読めないページがあってもスキップ
            continue

    full_text = "\n".join(texts)
    # プロンプト長が爆発しないように上限をかける（必要に応じて調整）
    return full_text[:20000]


# =========================
#  LLM 呼び出しロジック
# =========================

def call_llm_for_criteria(
    training_name: str,
    items: List[str],
    pdf_text: str,
    trainer_feedback: str,
    prompt_template: str,
    model: str = "gpt-4.1",
) -> Dict[str, Any]:
    """
    採点項目リストと PDF テキストなどから、評価基準 JSON を生成する。
    研修名・項目リスト・テキストなどは全て変数として差し込む。
    """
    item_list_text = "\n".join([f"- {name}" for name in items])

    # format() だと {training_name} 等以外の { } にも反応してエラーになるので、
    # 安全のため単純な replace() で埋め込む
    user_prompt = (
        prompt_template
        .replace("{training_name}", training_name or "")
        .replace("{item_list_text}", item_list_text)
        .replace("{pdf_text_chunk}", pdf_text)
        .replace("{trainer_feedback}", trainer_feedback or "")
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは企業研修の採点基準を設計する専門アナリストです。"
                    "出力は必ず有効な JSON のみとし、日本語で記述してください。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    # JSON パース（万一前後に余計な文字が混ざった場合に備えて簡易防御）
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start : end + 1]
            data = json.loads(json_str)
        else:
            raise ValueError("LLM の応答から JSON を抽出できませんでした。")

    return data


# =========================
#  JSON ↔ DataFrame 変換
# =========================

def json_to_criteria_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    LLM から返ってきた JSON を編集しやすい DataFrame に変換。
    行：項目 × 基準
    列：item_name, criterion_id, caption, description
    """
    rows = []
    for item in data.get("items", []):
        item_name = item.get("name", "")
        for crit in item.get("criteria", []):
            rows.append(
                {
                    "item_name": item_name,
                    "criterion_id": crit.get("id"),
                    "caption": crit.get("caption", ""),
                    "description": crit.get("description", ""),
                }
            )
    return pd.DataFrame(rows)


def update_json_from_df(data: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    DataFrame で編集された内容を元の JSON 構造に反映する。
    """
    key_to_row: Dict[tuple, pd.Series] = {}
    for _, row in df.iterrows():
        key = (row["item_name"], int(row["criterion_id"]))
        key_to_row[key] = row

    for item in data.get("items", []):
        item_name = item.get("name", "")
        for crit in item.get("criteria", []):
            key = (item_name, int(crit.get("id")))
            if key in key_to_row:
                row = key_to_row[key]
                crit["caption"] = row["caption"]
                crit["description"] = row["description"]

    return data

# =========================
#  研修紹介ページ参照
# =========================

#######################################################################
# ① HTML から原文そのままの項目を抽出（改善版）
#######################################################################

def extract_program_items_from_html(html: str) -> List[str]:
    """
    研修紹介ページの HTML から「プログラム」セクションの項目名をできるだけそのまま抜き出す。
    主に <h2>プログラム</h2> の直後の <ul><li> や <table> の行を想定。
    """
    soup = BeautifulSoup(html, "html.parser")

    # 「プログラム」という見出しを探す
    program_header = soup.find(
        lambda tag: tag.name in ["h1", "h2", "h3", "h4"]
        and "プログラム" in tag.get_text()
    )

    items: List[str] = []

    if program_header:
        # 見出し以降を順に走査
        for sib in program_header.find_all_next():
            # 別の大見出しまで来たら終了
            if sib.name in ["h1", "h2", "h3"] and sib is not program_header:
                break

            # <ul><li>
            if sib.name == "ul":
                for li in sib.find_all("li", recursive=False):
                    text = li.get_text(separator=" ", strip=True)
                    if text:
                        items.append(text)
                if items:
                    return items

            # <table>
            if sib.name == "table":
                for tr in sib.find_all("tr"):
                    cells = [c.get_text(separator=" ", strip=True) for c in tr.find_all(["th", "td"])]
                    if not cells:
                        continue
                    # 2列目がタイトルの場合が多い
                    if len(cells) >= 2:
                        text = cells[1]
                    else:
                        text = cells[0]
                    text = text.strip()
                    if text:
                        items.append(text)
                if items:
                    return items

    return []


#######################################################################
# ② HTML が取れなかった場合：ページ全文から LLM で項目抽出（原文優先）
#######################################################################

def extract_program_items_with_llm(page_text: str) -> List[str]:
    """
    ページ全体のテキストから LLM を使って「プログラム項目名」だけを抽出。
    """
    prompt = f"""
あなたは企業研修のプログラム構成を整理する専門アナリストです。
以下のページテキストから、「研修プログラムの項目名」だけを抽出してください。

- 各行1項目
- 時間（例：13:00〜）などは除外
- 内容タイトル（例：○○の理解・○○の実践 など）を優先
- 原文表現をできるだけ維持する
- 7〜10項目程度でまとめる
- 出力は「項目名のみ」を改行区切りで並べる

---
ページ全文:
\"\"\"{page_text}\"\"\"
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "あなたは研修プログラム構造の抽出に特化したアナリストです。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw = resp.choices[0].message.content or ""
    lines = [ln.strip(" ・-　") for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


#######################################################################
# ③ 最終整理：7項目（6〜8項目許容）MECE化、原文優先
#######################################################################

def normalize_program_items_with_llm(raw_items: List[str]) -> List[str]:
    """
    原文項目 raw_items をベースに、
    ・基本 8 項目（7〜9 も許容）
    ・原文の言葉をできるだけ優先
    ・もれなくダブりなく（MECE）
    ・各行1項目で7行に整形
    という要件で再編成する。
    """

    prompt = f"""
あなたは企業研修プログラムの構造化専門アナリストです。

以下は研修ページから抽出した「プログラム項目の原文リスト」です。

【原文リスト】
{chr(10).join("- " + item for item in raw_items)}

---

# 指示
1. 原文の表現をできるだけ維持しつつ、意味が重複する項目は統合する。
2. 細かすぎる項目は「意味の塊」にまとめてよい。
3. 基本的に **7項目** にまとめる（どうしても無理なら **6〜8項目** でも可）。
4. もれなく・ダブりなく（MECE）に再編成すること。
5. 出力は「各行1項目」、先頭に「・」を付けて縦に並べること。
6. できるだけ端的な言葉で短くまとめること。（15文字以内）

---

# 出力形式（厳守）
・項目1
・項目2
・項目3
・項目4
・項目5
・項目6
・項目7
・項目8
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "あなたは企業研修の内容整理に精通した専門アナリストです。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )

    lines = resp.choices[0].message.content.splitlines()
    results = [
        line.replace("・", "").strip(" -　")
        for line in lines
        if line.strip()
    ]
    return results


#######################################################################
# ④ URL 全体処理：HTML → 原文抽出 → LLM で 7 項目に MECE 化
#######################################################################

def get_program_items_from_url(url: str) -> List[str]:
    """
    URL からページを取得し、
    1. HTML構造から原文項目を抽出
    2. LLMが補完（fallback）
    3. 最後に 7 項目（6〜8可）MECE に統合
    の 3 段階で項目リストを返す。
    """

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    html = resp.text

    # ① HTML 構造から原文を抽出
    raw_items = extract_program_items_from_html(html)

    # ② HTML で取れない場合 → ページ全文から LLM 抽出
    if not raw_items:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        raw_items = extract_program_items_with_llm(text)

    # ③ 最終：7 項目（MECE）に正規化
    normalized_items = normalize_program_items_with_llm(raw_items)

    return normalized_items


# =========================
#  Excel 生成
# =========================

def generate_evaluation_excel(data: Dict[str, Any]) -> bytes:
    """
    評価表（講師・管理者向け）の Excel を生成。
    シート: 評価基準
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df = json_to_criteria_df(data)
        df.to_excel(writer, sheet_name="評価基準", index=False)
    return output.getvalue()


def generate_summary_sheet_excel(data: Dict[str, Any]) -> bytes:
    """
    受講者向けまとめシートの Excel を生成。
    添付画像のように、上部に期待欄＋
    各項目ごとに大きな記入欄を持つレイアウトにする。
    """
    output = io.BytesIO()

    # pandas は使うが、実際には xlsxwriter を直接操作する
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("まとめシート")
        writer.sheets["まとめシート"] = worksheet

        # 書式定義
        border = {"border": 1}
        bold = workbook.add_format({**border, "bold": True, "valign": "top"})
        normal = workbook.add_format({**border, "valign": "top", "text_wrap": True})

        # 列幅調整（お好みで調整してください）
        worksheet.set_column("A:A", 5)   # 番号など
        worksheet.set_column("B:B", 30)  # 項目名
        worksheet.set_column("C:C", 80)  # 記入欄

        row = 0  # 0-index

        # ===== 上部の共通欄（例：会社・上司からの期待など） =====
        # 1. 会社／上司からの期待
        worksheet.merge_range(row, 0, row, 2, "【会社または上司からの受講者への期待】", bold)
        row += 1
        worksheet.merge_range(row, 0, row, 2, "", normal)
        worksheet.set_row(row, 80)  # 記入しやすいように高さを確保
        row += 2  # 1 行空けたいなら +2 にする

        # 2. 受講に対する事前期待
        worksheet.merge_range(row, 0, row, 2, "【受講に対する事前期待（受講者による記入）】", bold)
        row += 1
        worksheet.merge_range(row, 0, row, 2, "", normal)
        worksheet.set_row(row, 80)
        row += 2

        # 3. 研修当日の振り返り欄（必要に応じて文言調整）
        worksheet.merge_range(row, 0, row, 2, "【研修当日に記入】", bold)
        row += 1
        worksheet.merge_range(row, 0, row, 2, "", normal)
        worksheet.set_row(row, 80)
        row += 2

        # ===== 各採点項目ごとの記入欄 =====
        items = data.get("items", [])

        for idx, item in enumerate(items, start=1):
            item_name = item.get("name", "")

            # summary_questions をヒントとしてセル内に表示したければここで結合
            questions = item.get("summary_questions", [])
            hint_text = "\n".join(f"・{q}" for q in questions) if questions else ""

            # 項目タイトル行：A〜C を結合
            worksheet.merge_range(row, 0, row, 2, f"{idx}. {item_name}", bold)
            row += 1

            # 記入欄（複数行分まとめて結合）
            # ここでは 4 行分(A〜C)を結合して大きな入力欄を作る
            start_row = row
            end_row = row + 3  # 行数はお好みで
            worksheet.merge_range(start_row, 0, end_row, 2, hint_text, normal)

            # 高さを少し大きめに
            for r in range(start_row, end_row + 1):
                worksheet.set_row(r, 50)

            row = end_row + 2  # 少し間隔を空ける

    return output.getvalue()



# =========================
#  デフォルトプロンプト（内容はそのまま）
#  ※ 研修名・項目・PDFテキストは {training_name} などで差し込み
# =========================

DEFAULT_PROMPT_TEMPLATE = """
## 研修情報
- 研修名: {training_name}

## 採点項目リスト
あなたは研修システムの採点基準を設計する専門アナリストです。以下の採点項目ごとに評価基準を作成してください:
{item_list_text}

## PDFテキスト（抽出要点）
以下は PDF から抽出したテキストです。基準作成の根拠として積極的に活用してください。
{pdf_text_chunk}


## 過去の講師フィードバック（任意）
{trainer_feedback}

---

# 必ず守る制約（キャプションと基準文の強化）

### ▼【採点基準】について
- 「短い要約」ではなく **その基準が判定する能力・行動の核心を 1 行で示す明確なフレーズ** にする。
- 抽象語ではなく **行動ベースの表現** を含める。
- 例：「現状と理想の差分を言語化し、改善行動を示している」

### ▼【説明文】について（最重要）
説明文は **最低 200〜300 文字程度** で記述し、以下の要素を必ず含める：

1. **Yes/No で判定できる具体条件**
2. **受講者の記述に表れる行動・思考の例（良い例）を 2 個以上**
3. **不十分な記述（悪い例）の具体例を 1 個以上**
4. **可能であれば PDF 内表現の引用を含める**
5. **抽象語（主体性・協働性など）を避け、観察可能な行動で記述する**

例：
- 良い例：「管理者とは何かを自分の言葉で述べ、責任範囲と求められる役割を整理している」
- 悪い例：「役割が大事だと書くだけで、自分がどう行動するかが書かれていない」

---

#  最終出力要件（JSON 固定）

必ず以下の JSON 構造のみを出力してください：

{
  "training_name": "....",
  "items": [
    {
      "name": "採点項目名",
      "description": "この項目の要点・狙い（100文字程度、日本語）",
      "criteria": [
        {
          "id": 1,
          "caption": "基準1の要点（1行程度）",
          "description": "基準1の詳細な説明。Yes/Noで判定可能。良い記述の具体例と不十分な記述の例を含める。"
        },
        {
          "id": 2,
          "caption": "基準2の要点（1行程度）",
          "description": "基準2の詳細な説明。Yes/Noで判定可能。良い記述の具体例と不十分な記述の例を含める。"
        },
        {
          "id": 3,
          "caption": "基準3の要点（1行程度）",
          "description": "基準3の詳細な説明。Yes/Noで判定可能。良い記述の具体例と不十分な記述の例を含める。"
        },
        {
          "id": 4,
          "caption": "基準4の要点（1行程度）",
          "description": "基準4の詳細な説明。Yes/Noで判定可能。良い記述の具体例と不十分な記述の例を含める。"
        },
        {
          "id": 5,
          "caption": "基準5の要点（1行程度）",
          "description": "基準5の詳細な説明。Yes/Noで判定可能。良い記述の具体例と不十分な記述の例を含める。"
        }
      ],
      "summary_questions": [
        "まとめシート用の設問文1（日本語）",
        "必要なら設問文2"
      ]
    }
  ]
}


JSON 以外のテキストは出力しない。
"""


# =========================
#  Streamlit アプリ本体
# =========================

def main():
    st.set_page_config(page_title="採点基準作成システム β", layout="wide")
    st.title("採点基準作成システム β版")

    # セッション状態初期化
    if "criteria_json" not in st.session_state:
        st.session_state["criteria_json"] = None

    tab1, tab2, tab3 = st.tabs(["① インプット設定", "② 基準編集", "③ Excel出力"])

    # ---- タブ1：インプット設定（研修名・PDF・採点項目などすべて変数）----
    with tab1:
        st.subheader("研修情報・インプット設定")

        # 研修名
        training_name = st.text_input(
            "研修名",
            value="",
            placeholder="例：新入社員実務基本2日間コース／管理職研修 など",
        )

        # ① プログラムURL入力欄
        program_url = st.text_input(
            "研修プログラムのURL（JMAサイトなど）",
            value="",
            placeholder="例：https://school.jma.or.jp/products/detail.php?product_id=100157",
        )

        # session_state に採点項目リストを保持
        if "item_list_text" not in st.session_state:
            st.session_state["item_list_text"] = ""

        # ② URLから自動抽出ボタン
        if st.button("URLから採点項目を自動抽出"):
            if not program_url.strip():
                st.error("まず研修プログラムのURLを入力してください。")
            else:
                try:
                    with st.spinner("Webページからプログラム項目を抽出しています..."):
                        items = get_program_items_from_url(program_url.strip())
                    if not items:
                        st.warning("プログラム項目を抽出できませんでした。手動で入力してください。")
                    else:
                        st.session_state["item_list_text"] = "\n".join(items)
                        st.success(f"{len(items)}件の項目を抽出しました。下の採点項目リストに反映しています。")
                except Exception as e:
                    st.error(f"項目抽出でエラーが発生しました: {e}")

        # ③ 採点項目リスト（手動編集も可能）
        item_list_text = st.text_area(
            "採点項目リスト（1行1項目）",
            height=200,
            key="item_list_text",
            help="URLから自動抽出した項目をベースに、ここで手動で修正・追記できます。",
        )

        # ④ 研修テキスト PDF
        uploaded_pdf = st.file_uploader(
            "研修テキスト PDF をアップロード（どの研修でも可）",
            type=["pdf"],
        )

        # ⑤ 任意の講師フィードバック
        trainer_feedback = st.text_area(
            "過去の講師フィードバック・評価観点（任意）",
            height=150,
            placeholder=(
                "例：\n"
                "・この研修では〇〇の行動変容を重視したい\n"
                "・振り返りでは具体的なエピソードが書けているかを見たい など"
            ),
        )

        # ⑥ プロンプトテンプレート
        prompt_template = st.text_area(
            "基準生成用プロンプトテンプレート",
            value=DEFAULT_PROMPT_TEMPLATE,
            height=260,
        )

        # ⑦ モデル名
        model_name = st.text_input("使用モデル名", value="gpt-4.1")

        # ⑧ 基準案生成ボタン
        if st.button("基準案を生成", type="primary"):
            if uploaded_pdf is None:
                st.error("PDF をアップロードしてください。")
            elif not training_name.strip():
                st.error("研修名を入力してください。")
            else:
                with st.spinner("PDF を解析して基準案を生成しています..."):
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    items = [
                        line.strip()
                        for line in item_list_text.splitlines()
                        if line.strip()
                    ]

                    try:
                        data = call_llm_for_criteria(
                            training_name=training_name,
                            items=items,
                            pdf_text=pdf_text,
                            trainer_feedback=trainer_feedback,
                            prompt_template=prompt_template,
                            model=model_name,
                        )
                    except Exception as e:
                        st.error(f"LLM 呼び出しでエラーが発生しました: {e}")
                    else:
                        st.session_state["criteria_json"] = data
                        st.success("基準案を生成しました。「② 基準編集」タブで確認できます。")

    # ---- タブ2：基準編集 ----
    with tab2:
        st.subheader("基準編集")

        data = st.session_state.get("criteria_json")
        if data is None:
            st.info("まだ基準案が生成されていません。「① インプット設定」から生成してください。")
        else:
            df = json_to_criteria_df(data)
            item_names = sorted(df["item_name"].unique())

            selected_item = st.selectbox("編集する採点項目を選択", options=item_names)
            df_item = df[df["item_name"] == selected_item].copy()

            st.markdown(f"**{selected_item} の基準（5つ）**")
            edited_df = st.data_editor(df_item, num_rows="dynamic", key="editor_df")

            if st.button("この項目の変更を反映"):
                df.update(edited_df)
                updated_data = update_json_from_df(data, df)
                st.session_state["criteria_json"] = updated_data
                st.success("基準 JSON を更新しました。")

            with st.expander("JSON 全体（デバッグ用）"):
                st.json(st.session_state["criteria_json"])

    # ---- タブ3：Excel 出力 ----
    with tab3:
        st.subheader("Excel 出力")

        data = st.session_state.get("criteria_json")
        if data is None:
            st.info("まだ基準案が生成されていません。")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 評価表（講師・管理者向け）")
                eval_bytes = generate_evaluation_excel(data)
                st.download_button(
                    label="評価基準 Excel をダウンロード",
                    data=eval_bytes,
                    file_name="evaluation_criteria.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

            with col2:
                st.markdown("### 受講者向けまとめシート")
                summary_bytes = generate_summary_sheet_excel(data)
                st.download_button(
                    label="まとめシート Excel をダウンロード",
                    data=summary_bytes,
                    file_name="summary_sheet.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )


if __name__ == "__main__":
    main()
