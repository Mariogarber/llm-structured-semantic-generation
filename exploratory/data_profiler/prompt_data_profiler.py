import re
import html
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class PromptDataProfiler:
    """
    Perfilador de prompts para datasets estructurados.
    
    Busca archivos:
      - question.txt
      - question_simplified.txt
    
    Estructura esperada aproximada:
      root/
        data/
          question/
            Envoy/
              q1/
                question.txt
                question_simplified.txt
              q2/
                ...
            OtroServicio/
              ...
    """

    DEFAULT_STOPWORDS = {
        "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at", "by",
        "with", "from", "as", "is", "are", "be", "this", "that", "these", "those",
        "it", "its", "into", "than", "then", "your", "you", "use", "using",
        "create", "generate", "write", "make", "define", "set", "add",
        "de", "la", "el", "los", "las", "un", "una", "y", "o", "en", "para",
        "con", "del", "por", "que", "se", "al", "como"
    }

    def __init__(
        self,
        root_dir: str,
        output_html: str = "prompt_profile_report.html",
        question_dir_parts: Optional[List[str]] = None,
        include_simplified: bool = True,
        min_token_len: int = 2,
        stopwords: Optional[set] = None,
    ):
        self.root_dir = Path(root_dir)
        self.output_html = output_html
        self.question_dir_parts = question_dir_parts or ["data", "question"]
        self.include_simplified = include_simplified
        self.min_token_len = min_token_len
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS

        self.df: Optional[pd.DataFrame] = None

    # =========================
    # Lectura del dataset
    # =========================
    def load_prompts(self) -> pd.DataFrame:
        """
        Recorre los directorios y carga prompts encontrados.
        """
        question_root = self.root_dir.joinpath(*self.question_dir_parts)

        if not question_root.exists():
            raise FileNotFoundError(f"No existe la ruta base de questions: {question_root}")

        rows = []

        for service_dir in sorted(question_root.iterdir()):
            if not service_dir.is_dir():
                continue

            service_name = service_dir.name

            for q_dir in sorted(service_dir.iterdir()):
                if not q_dir.is_dir():
                    continue

                q_id = q_dir.name

                candidate_files = [("question", q_dir / "question.txt")]
                if self.include_simplified:
                    candidate_files.append(("question_simplified", q_dir / "question_simplified.txt"))

                for variant, file_path in candidate_files:
                    if file_path.is_file():
                        with open(file_path, "r", encoding="utf-8") as f:
                            prompt = f.read()

                        rows.append({
                            "service": service_name,
                            "question_id": q_id,
                            "variant": variant,
                            "file_path": str(file_path),
                            "prompt": prompt
                        })

        if not rows:
            raise ValueError(f"No se encontraron prompts en {question_root}")

        df = pd.DataFrame(rows)
        self.df = self._enrich_features(df)
        return self.df

    # =========================
    # Feature engineering
    # =========================
    def _enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["prompt_clean"] = df["prompt"].fillna("").astype(str)
        df["char_count"] = df["prompt_clean"].apply(len)
        df["line_count"] = df["prompt_clean"].apply(lambda x: x.count("\n") + 1 if x else 0)
        df["sentence_count"] = df["prompt_clean"].apply(self._count_sentences)
        df["tokens"] = df["prompt_clean"].apply(self._tokenize)
        df["word_count"] = df["tokens"].apply(len)
        df["unique_word_count"] = df["tokens"].apply(lambda toks: len(set(toks)))
        df["lexical_diversity"] = df.apply(
            lambda row: row["unique_word_count"] / row["word_count"] if row["word_count"] > 0 else 0.0,
            axis=1
        )
        df["avg_word_length"] = df["tokens"].apply(
            lambda toks: np.mean([len(t) for t in toks]) if toks else 0.0
        )
        df["avg_sentence_length_words"] = df.apply(
            lambda row: row["word_count"] / row["sentence_count"] if row["sentence_count"] > 0 else 0.0,
            axis=1
        )
        df["digit_ratio"] = df["prompt_clean"].apply(self._digit_ratio)
        df["uppercase_ratio"] = df["prompt_clean"].apply(self._uppercase_ratio)
        df["question_mark_count"] = df["prompt_clean"].apply(lambda x: x.count("?"))
        df["colon_count"] = df["prompt_clean"].apply(lambda x: x.count(":"))
        df["comma_count"] = df["prompt_clean"].apply(lambda x: x.count(","))
        df["bullet_like_count"] = df["prompt_clean"].apply(self._bullet_like_count)

        return df

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9_\-/\.]+\b", text)
        tokens = [t for t in tokens if len(t) >= self.min_token_len]
        return tokens

    def _count_sentences(self, text: str) -> int:
        if not text.strip():
            return 0
        parts = re.split(r"[.!?\n]+", text)
        return len([p for p in parts if p.strip()])

    def _digit_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        digit_count = sum(ch.isdigit() for ch in text)
        return digit_count / len(text)

    def _uppercase_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        alpha = [ch for ch in text if ch.isalpha()]
        if not alpha:
            return 0.0
        upper = sum(ch.isupper() for ch in alpha)
        return upper / len(alpha)

    def _bullet_like_count(self, text: str) -> int:
        lines = text.splitlines()
        pattern = re.compile(r"^\s*([-*•]|\d+\.)\s+")
        return sum(bool(pattern.match(line)) for line in lines)

    # =========================
    # Resúmenes agregados
    # =========================
    def _summary_table(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Primero debes ejecutar load_prompts()")

        numeric_cols = [
            "char_count", "word_count", "line_count", "sentence_count",
            "unique_word_count", "lexical_diversity", "avg_word_length",
            "avg_sentence_length_words", "digit_ratio", "uppercase_ratio",
            "question_mark_count", "colon_count", "comma_count", "bullet_like_count"
        ]

        summary = self.df[numeric_cols].describe().T.reset_index()
        summary = summary.rename(columns={"index": "metric"})
        return summary

    def _service_counts(self) -> pd.DataFrame:
        return (
            self.df.groupby(["service", "variant"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "service"], ascending=[False, True])
        )

    def _top_words(self, top_n: int = 30) -> pd.DataFrame:
        tokens = []
        for toks in self.df["tokens"]:
            tokens.extend([
                t for t in toks
                if t not in self.stopwords and not t.isdigit()
            ])

        counts = Counter(tokens)
        data = counts.most_common(top_n)
        return pd.DataFrame(data, columns=["word", "count"])

    def _top_bigrams(self, top_n: int = 25) -> pd.DataFrame:
        bigrams = []
        for toks in self.df["tokens"]:
            filtered = [t for t in toks if t not in self.stopwords]
            for i in range(len(filtered) - 1):
                bigrams.append(f"{filtered[i]} {filtered[i+1]}")

        counts = Counter(bigrams)
        data = counts.most_common(top_n)
        return pd.DataFrame(data, columns=["bigram", "count"])

    def _extreme_examples(self) -> Dict[str, Dict]:
        shortest = self.df.loc[self.df["word_count"].idxmin()].to_dict()
        longest = self.df.loc[self.df["word_count"].idxmax()].to_dict()
        return {"shortest": shortest, "longest": longest}

    # =========================
    # Gráficas
    # =========================
    def _fig_prompt_count_by_service(self) -> str:
        service_counts = self._service_counts()
        fig = px.bar(
            service_counts,
            x="service",
            y="count",
            color="variant",
            barmode="group",
            title="Número de prompts por servicio y variante"
        )
        fig.update_layout(height=500)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_word_count_hist(self) -> str:
        fig = px.histogram(
            self.df,
            x="word_count",
            color="variant",
            marginal="box",
            nbins=40,
            title="Distribución de longitud de prompts (número de palabras)"
        )
        fig.update_layout(height=500)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_char_count_hist(self) -> str:
        fig = px.histogram(
            self.df,
            x="char_count",
            color="variant",
            marginal="box",
            nbins=40,
            title="Distribución de longitud de prompts (número de caracteres)"
        )
        fig.update_layout(height=500)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_lexical_diversity(self) -> str:
        fig = px.box(
            self.df,
            x="variant",
            y="lexical_diversity",
            color="variant",
            points="all",
            title="Diversidad léxica por variante"
        )
        fig.update_layout(height=500)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_avg_sentence_length(self) -> str:
        fig = px.box(
            self.df,
            x="variant",
            y="avg_sentence_length_words",
            color="variant",
            points="all",
            title="Longitud media de frase por variante"
        )
        fig.update_layout(height=500)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_words_bar(self, top_n: int = 30) -> str:
        top_words = self._top_words(top_n=top_n)
        fig = px.bar(
            top_words.iloc[::-1],
            x="count",
            y="word",
            orientation="h",
            title=f"Top {top_n} palabras más frecuentes"
        )
        fig.update_layout(height=700)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_bigrams_bar(self, top_n: int = 25) -> str:
        top_bigrams = self._top_bigrams(top_n=top_n)
        fig = px.bar(
            top_bigrams.iloc[::-1],
            x="count",
            y="bigram",
            orientation="h",
            title=f"Top {top_n} bigramas más frecuentes"
        )
        fig.update_layout(height=700)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_scatter_words_vs_lexical_diversity(self) -> str:
        fig = px.scatter(
            self.df,
            x="word_count",
            y="lexical_diversity",
            color="variant",
            hover_data=["service", "question_id", "file_path"],
            title="Longitud del prompt vs diversidad léxica"
        )
        fig.update_layout(height=550)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_service_vs_wordcount(self) -> str:
        fig = px.box(
            self.df,
            x="service",
            y="word_count",
            color="variant",
            points="outliers",
            title="Longitud del prompt por servicio"
        )
        fig.update_layout(height=600)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _fig_correlation_heatmap(self) -> str:
        numeric_cols = [
            "char_count", "word_count", "line_count", "sentence_count",
            "unique_word_count", "lexical_diversity", "avg_word_length",
            "avg_sentence_length_words", "digit_ratio", "uppercase_ratio",
            "question_mark_count", "colon_count", "comma_count", "bullet_like_count"
        ]
        corr = self.df[numeric_cols].corr(numeric_only=True)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hoverongaps=False
            )
        )
        fig.update_layout(title="Matriz de correlación entre métricas del prompt", height=700)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    # =========================
    # HTML report
    # =========================
    def generate_html_report(self) -> str:
        if self.df is None:
            self.load_prompts()

        summary_df = self._summary_table()
        extremes = self._extreme_examples()

        summary_html = summary_df.to_html(index=False, classes="table table-striped table-sm", border=0)
        sample_table_html = (
            self.df[["service", "question_id", "variant", "word_count", "char_count", "lexical_diversity", "file_path"]]
            .sort_values(["service", "question_id", "variant"])
            .to_html(index=False, classes="table table-striped table-sm", border=0)
        )

        shortest_prompt_html = self._format_prompt_card(extremes["shortest"], title="Prompt más corto")
        longest_prompt_html = self._format_prompt_card(extremes["longest"], title="Prompt más largo")

        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Prompt Profile Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background-color: #f8f9fa;
            padding: 20px;
        }}
        .tab-content {{
            margin-top: 20px;
        }}
        .metric-card {{
            border-radius: 14px;
            padding: 16px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 16px;
        }}
        .prompt-box {{
            white-space: pre-wrap;
            background: #f1f3f5;
            padding: 12px;
            border-radius: 10px;
            font-family: Consolas, monospace;
            font-size: 0.92rem;
        }}
        .plot-container {{
            background: white;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}
        .table {{
            background: white;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-4">Informe de perfilado de prompts</h1>

        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Total prompts</h5>
                    <p class="fs-3">{len(self.df)}</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Servicios</h5>
                    <p class="fs-3">{self.df['service'].nunique()}</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Variantes</h5>
                    <p class="fs-3">{self.df['variant'].nunique()}</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Media palabras</h5>
                    <p class="fs-3">{self.df['word_count'].mean():.2f}</p>
                </div>
            </div>
        </div>

        <ul class="nav nav-tabs" id="reportTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#overview" type="button">Overview</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#lengths" type="button">Longitudes</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#lexical" type="button">Léxico</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#services" type="button">Servicios</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#correlations" type="button">Correlaciones</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#samples" type="button">Ejemplos</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#table" type="button">Tabla</button>
            </li>
        </ul>

        <div class="tab-content">
            <div class="tab-pane fade show active" id="overview">
                <div class="plot-container">{self._fig_prompt_count_by_service()}</div>
                <div class="metric-card">
                    <h4>Resumen estadístico</h4>
                    {summary_html}
                </div>
            </div>

            <div class="tab-pane fade" id="lengths">
                <div class="plot-container">{self._fig_word_count_hist()}</div>
                <div class="plot-container">{self._fig_char_count_hist()}</div>
                <div class="plot-container">{self._fig_avg_sentence_length()}</div>
            </div>

            <div class="tab-pane fade" id="lexical">
                <div class="plot-container">{self._fig_lexical_diversity()}</div>
                <div class="plot-container">{self._fig_scatter_words_vs_lexical_diversity()}</div>
                <div class="plot-container">{self._fig_words_bar(30)}</div>
                <div class="plot-container">{self._fig_bigrams_bar(25)}</div>
            </div>

            <div class="tab-pane fade" id="services">
                <div class="plot-container">{self._fig_service_vs_wordcount()}</div>
            </div>

            <div class="tab-pane fade" id="correlations">
                <div class="plot-container">{self._fig_correlation_heatmap()}</div>
            </div>

            <div class="tab-pane fade" id="samples">
                <div class="row">
                    <div class="col-md-6">{shortest_prompt_html}</div>
                    <div class="col-md-6">{longest_prompt_html}</div>
                </div>
            </div>

            <div class="tab-pane fade" id="table">
                <div class="metric-card">
                    <h4>Tabla de prompts cargados</h4>
                    {sample_table_html}
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        with open(self.output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        return self.output_html

    def _format_prompt_card(self, row: Dict, title: str) -> str:
        prompt = html.escape(row["prompt"])
        return f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <p><strong>Service:</strong> {row['service']}</p>
            <p><strong>Question ID:</strong> {row['question_id']}</p>
            <p><strong>Variant:</strong> {row['variant']}</p>
            <p><strong>Words:</strong> {row['word_count']} | <strong>Chars:</strong> {row['char_count']}</p>
            <p><strong>Path:</strong> {html.escape(row['file_path'])}</p>
            <div class="prompt-box">{prompt[:5000]}</div>
        </div>
        """

    # =========================
    # Pipeline principal
    # =========================
    def run(self) -> pd.DataFrame:
        df = self.load_prompts()
        self.generate_html_report()
        return df