from dataclasses import dataclass
from typing import Any, Iterable, Optional
import os
from loguru import logger

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None


@dataclass
class LLMConfig:
    model_path: str
    n_ctx: int = 2048
    n_threads: int = max(os.cpu_count() or 4, 4)
    n_gpu_layers: int = 0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 256
    system_prompt: str = (
        "Tu es un assistant vocal embarqué sur un Jetson Orin Nano. "
        "Réponds en français, de façon concise, claire, et utile. "
        "Si tu ne sais pas, dis-le"
        "les lignes sont de la forme : "
        "Comité IA;Projet;Technique IA;Framework;Description courte;Rôle;Personne;Poste;Equipe;Période;Version en DEV;Version en OPE;Objectif;"
        "les différentes lignes sont séparées par des |"
    )


class LLMService:
    def __init__(self, config: LLMConfig):
        self.cfg = config
        self.backend = None

        if Llama is not None:
            self.backend = "llama_cpp"
            if not os.path.isfile(config.model_path):
                raise FileNotFoundError(f"Modèle GGUF introuvable: {config.model_path}")
            self.llm = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_threads=self.cfg.n_threads,
                n_gpu_layers=self.cfg.n_gpu_layers,
                verbose=False,
            )
        elif GPT4All is not None:
            self.backend = "gpt4all"
            # GPT4All gère les chemins différemment, souvent juste le nom du fichier ou dossier
            abs_path = os.path.abspath(config.model_path)
            model_dir = os.path.dirname(abs_path)
            model_name = os.path.basename(abs_path)

            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Dossier de modèle introuvable : {model_dir}")

            # Message informatif pour l'utilisateur s'il n'a pas le fichier
            if not os.path.isfile(abs_path) and not os.path.isfile(
                os.path.join(model_dir, model_name)
            ):
                logger.warning(
                    f"ATTENTION: Le modèle '{model_name}' semble absent de '{model_dir}'. GPT4All va peut-être échouer."
                )

            self.llm = GPT4All(model_name, model_path=model_dir, allow_download=False)
        else:
            raise ImportError(
                "Aucun backend LLM trouvé. Installez 'llama-cpp-python' ou 'gpt4all'."
            )

    def _format_docs(self, docs: str) -> str:
        if not docs:
            return ""

    def generate(self, question: str, context) -> str:
        question = (question or "").strip()
        if not question:
            return "Je n'ai pas reçu de question."

        # context = self._format_docs(context)
        user_content = (
            f"Question: {question}\n"
            + (f"\nContexte:\n{context}\n" if context else "\n")
            + "\nRéponds maintenant."
        )

        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_content},
        ]

        if self.backend == "llama_cpp":
            out = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_tokens,
            )
            try:
                return (
                    out["choices"][0]["message"]["content"].strip()
                    # + "context : "
                    # + context
                )
            except Exception:
                return str(out)

        elif self.backend == "gpt4all":
            # GPT4All chat connection is stateful usually, but generate method is simpler?
            # Actually GPT4All python bindings have .chat_session() or .generate().
            # Let's use simple .generate() but it's checking prompt format.
            # Better to use chat_session context manager for chat format.
            with self.llm.chat_session(self.cfg.system_prompt):
                response = self.llm.generate(
                    user_content,
                    max_tokens=self.cfg.max_tokens,
                    temp=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )
                return response
                # + "context : " + context

        try:
            return (
                out["choices"][0]["message"]["content"].strip()
                #   + "context : " + context
            )
        except Exception:
            return str(out)

    def generate_with_system(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        if self.backend == "llama_cpp":
            out = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_tokens,
            )
            try:
                return out["choices"][0]["message"]["content"].strip()
            except Exception:
                return str(out)

        elif self.backend == "gpt4all":
            with self.llm.chat_session(system_prompt):
                return self.llm.generate(
                    user_content,
                    max_tokens=self.cfg.max_tokens,
                    temp=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )

        return ""

