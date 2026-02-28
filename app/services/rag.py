import pandas as pd
import sqlite3
from app.config import MAX_SQL
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self, csv_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        df = pd.read_csv(csv_path, sep=";")
        self.rows = df.astype(str).agg(";".join, axis=1).tolist()

        # df = df.fillna("")

        # # Prefix each cell with its column name (including empty strings)
        # for col in df.columns:
        #     df[col] = col + " : " + df[col].astype(str)

        # self.rows = df.astype(str).agg(";".join, axis=1).tolist()

        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode(self.rows, convert_to_numpy=True, normalize_embeddings=True)

    def top_k(self, query: str, k: int = 5):
        if not query:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        scores = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.rows[i], float(scores[i])) for i in top_idx]

    def format_context(self, query: str, k: int = 5) -> str:
        hits = self.top_k(query, k)
        return "".join([row + "|" for (row, _) in hits])

# class RAGService:
#     def __init__(self, csv_path: str, table_name: str = "projets_ia"):
#         df = pd.read_csv(csv_path, sep=";").fillna("")
#         df.columns = [self._normalize_column(c) for c in df.columns]
#         def _row_summary(row):
#             parts = []
#             if row["Comité_IA"] and row["Projet"]:
#                 parts.append(
#                     f"Le comite IA {row['Comité_IA']} est responsable du projet {row['Projet']}"
#                 )
#             if row["Période"]:
#                 parts.append(f"qui se deroule en {row['Période']}")
#             if row["Personne"] and row["Rôle"]:
#                 parts.append(
#                     f"{row['Personne']} y travaille en tant que {row['Rôle']}"
#                 )
#             if row["Poste"] and row["Equipe"]:
#                 parts.append(
#                     f"au poste de {row['Poste']} au sein de l'equipe {row['Equipe']}"
#                 )
#             if row["Technique_IA"] and row["Framework"]:
#                 parts.append(
#                     f"avec la technique IA {row['Technique_IA']} et le framework {row['Framework']}"
#                 )
#             if row["Description_courte"]:
#                 parts.append(
#                     f"la description courte du projet est {row['Description_courte']}"
#                 )
#             if row["Objectif"]:
#                 parts.append(f"l'objectif est {row['Objectif']}")
#             if row["Version_en_DEV"] and row["Version_en_OPE"]:
#                 parts.append(
#                     f"la version en DEV est {row['Version_en_DEV']} et la version en OPE est {row['Version_en_OPE']}"
#                 )
#             return ", ".join(parts) + "." if parts else ""

#         df["row_summary"] = df.apply(_row_summary, axis=1)
#         self.table_name = table_name
#         self.conn = sqlite3.connect(":memory:")
#         df.to_sql(self.table_name, self.conn, index=False, if_exists="replace")
#         self.columns = list(df.columns)

#     def schema_prompt(self) -> str:
#         cols = ", ".join([f'"{c}"' for c in self.columns])
#         return (
#             "Tu es un assistant SQL. Retourne uniquement une requete SQL SELECT. "
#             f"Table: {self.table_name}. Colonnes: {cols}. "
#             "Utilise des guillemets doubles pour les noms de colonnes. "
#             f"Ajoute LIMIT {MAX_SQL}. Pas d'explications."
#         )

#     def _normalize_column(self, name: str) -> str:
#         name = (name or "").strip()
#         name = name.replace(" ", "_")
#         name = name.replace(":", "_")
#         return name

#     def _safe_sql(self, sql: str) -> str:
#         sql_clean = (sql or "").strip()
#         if sql_clean.startswith("```"):
#             sql_clean = sql_clean.strip("`").strip()
#         sql_clean = sql_clean.replace(";", "")
#         lowered = sql_clean.lower()
#         if "select" not in lowered:
#             raise ValueError("Only SELECT queries are allowed.")
#         select_pos = lowered.find("select")
#         sql_clean = sql_clean[select_pos:].strip()
#         lowered = sql_clean.lower()
#         if not lowered.startswith("select"):
#             raise ValueError("Only SELECT queries are allowed.")
#         # block multi-statements
#         if ";" in sql_clean.rstrip(";"):
#             raise ValueError("Multiple statements are not allowed.")
#         sql_clean = self._force_row_summary(sql_clean)
#         if "limit" not in lowered:
#             sql_clean = f"{sql_clean} LIMIT {MAX_SQL}"
#         return sql_clean

#     def _force_row_summary(self, sql: str) -> str:
#         lowered = sql.lower()
#         select_pos = lowered.find("select")
#         from_pos = lowered.find("from ", select_pos)
#         if from_pos == -1:
#             return sql
#         select_clause = lowered[select_pos:from_pos]
#         if "row_summary" in select_clause:
#             return sql
#         alias = self._extract_first_alias(sql, from_pos)
#         if alias:
#             return f'SELECT {alias}."row_summary" {sql[from_pos:]}'
#         return f'SELECT "row_summary" {sql[from_pos:]}'

#     def _extract_first_alias(self, sql: str, from_pos: int) -> str:
#         tail = sql[from_pos + 5 :].strip()
#         tokens = tail.split()
#         if not tokens:
#             return ""
#         if len(tokens) >= 3 and tokens[1].lower() == "as":
#             return tokens[2].rstrip(",")
#         if len(tokens) >= 2:
#             if tokens[1].lower() in {
#                 "join",
#                 "where",
#                 "inner",
#                 "left",
#                 "right",
#                 "full",
#                 "cross",
#                 "on",
#                 "group",
#                 "order",
#                 "limit",
#             }:
#                 return ""
#             return tokens[1].rstrip(",")
#         return ""

#     def run_sql(self, sql: str) -> pd.DataFrame:
#         sql_clean = self._safe_sql(sql)
#         print(sql_clean)
#         return pd.read_sql_query(sql_clean, self.conn)

#     def format_context_from_sql(self, sql: str) -> str:
#         df = self.run_sql(sql)
#         if df.empty:
#             return ""
#         df = df.drop_duplicates()
#         rows = df.astype(str).agg(";".join, axis=1).tolist()
#         return "".join([row + "|" for row in rows])
