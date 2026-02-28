import gradio as gr
import requests
import base64
import soundfile as sf
import io
import tempfile
import os

FASTAPI_URL = "http://127.0.0.1:8000/voice/chat"

# CSS personnalisé - Thème moderne avec meilleure lisibilité


def send_audio(audio, progress=gr.Progress()):
    """
    audio = (sample_rate, numpy_array)
    Gradio fournit l'audio brut.
    On l'encode en WAV et on l'envoie au backend.
    """

    # Vérification de l'audio
    if audio is None:
        return (
            "⚠️ Aucun audio détecté",
            "❌ **Erreur** : Veuillez enregistrer un message vocal avant d'envoyer.",
            None,
            "🎤 En attente d'enregistrement...",
        )

    sr, data = audio

    # Étape 1 : Préparation
    progress(0.2, desc="📦 Préparation de l'audio...")

    # Conversion en WAV
    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV")
    buffer.seek(0)

    # Étape 2 : Envoi
    progress(0.4, desc="📤 Envoi au serveur...")

    files = {"audio": ("audio.wav", buffer, "audio/wav")}

    try:
        response = requests.post(FASTAPI_URL, files=files)  # , timeout=30

        # Étape 3 : Traitement
        progress(0.7, desc="🤖 Traitement de la réponse...")

        if response.status_code == 200:
            data = response.json()
            transcription = data.get("transcription", "")
            answer = data.get("answer", "")

            # Étape 4 : Audio
            progress(0.9, desc="🔊 Génération de l'audio...")

            audio_path = None
            audio_b64 = data.get("audio_reply")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                audio_path = tmp.name

            # Succès
            progress(1.0, desc="✅ Terminé !")

            status_msg = "✅ **Traitement réussi** - Réponse générée avec succès !"

            return (
                (
                    f"🎤 **Vous avez dit :** {transcription}"
                    if transcription
                    else "⚠️ Transcription vide"
                ),
                f"🤖 **Réponse :** {answer}" if answer else "⚠️ Pas de réponse générée",
                audio_path,
                status_msg,
            )
        else:
            error_msg = f"❌ **Erreur serveur** (Code {response.status_code})"
            return (
                "❌ Erreur de transcription",
                f"Le serveur a retourné une erreur. Détails : {response.text[:200]}",
                None,
                error_msg,
            )

    except requests.exceptions.Timeout:
        return (
            "⏱️ Délai dépassé",
            "❌ **Erreur** : Le serveur met trop de temps à répondre. Veuillez réessayer.",
            None,
            "⏱️ Timeout - Serveur trop lent",
        )
    except requests.exceptions.ConnectionError:
        return (
            "🔌 Connexion impossible",
            "❌ **Erreur** : Impossible de se connecter au serveur. Vérifiez qu'il est bien démarré.",
            None,
            "🔌 Serveur inaccessible",
        )
    except Exception as e:
        return (
            "💥 Erreur inattendue",
            f"❌ **Erreur** : {str(e)}",
            None,
            f"💥 Erreur : {type(e).__name__}",
        )


# Construction de l'interface
with gr.Blocks() as app:

    # En-tête
    # En-tête
    with gr.Row(elem_classes="main-header"):
        gr.Markdown(
            """
            # 🚗 Assistant Vocal Renault
            ### Votre assistant intelligent basé sur l'IA vocale
            Posez vos questions à voix haute et recevez des réponses instantanées !
            """
        )

    # Zone de statut
    status_box = gr.Textbox(
        value="🎤 Prêt à vous écouter - Enregistrez votre question",
        label="📊 Statut",
        interactive=False,
        container=True,
        elem_classes="status-box",
    )

    # Interface principale en 2 colonnes
    # Interface principale en 2 colonnes
    with gr.Row(elem_classes="column-container"):

        # Colonne gauche - Entrée
        # Colonne gauche - Entrée
        with gr.Column(scale=1, elem_classes="glass-card"):
            gr.Markdown("### 🎙️ Enregistrement")
            gr.Markdown("*Cliquez sur le micro pour commencer l'enregistrement*")

            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎤 Votre question vocale",
                interactive=True,
            )

            button = gr.Button(
                "🚀 Envoyer la question",
                variant="primary",
                size="lg",
                elem_classes="btn-primary",
            )

        # Colonne droite - Sortie
        # Colonne droite - Sortie
        with gr.Column(scale=1, elem_classes="glass-card"):
            gr.Markdown("### 💬 Résultats")

            with gr.Accordion("📝 Transcription (ce que vous avez dit)", open=True):
                transcription_out = gr.Textbox(
                    label="Texte transcrit",
                    placeholder="La transcription de votre audio apparaîtra ici...",
                    lines=3,
                )

            with gr.Accordion("🤖 Réponse de l'assistant", open=True):
                llm_out = gr.Textbox(
                    label="Réponse générée",
                    placeholder="La réponse de l'IA apparaîtra ici...",
                    lines=5,
                )

            with gr.Accordion("🔊 Audio de la réponse", open=True):
                tts_audio_out = gr.Audio(
                    label="Écouter la réponse",
                    type="filepath",
                    autoplay=False,  # Wait the user launch it because it's not a super quality
                )

    # Informations supplémentaires
    with gr.Accordion("ℹ️ Comment utiliser l'assistant", open=False):
        gr.Markdown(
            """
            ### Mode d'emploi :
            
            1. **🎤 Enregistrer** : Cliquez sur le microphone et parlez clairement
            2. **⏹️ Arrêter** : Cliquez à nouveau pour terminer l'enregistrement
            3. **🚀 Envoyer** : Cliquez sur le bouton "Envoyer la question"
            4. **⏳ Patienter** : L'IA traite votre demande
            5. **✅ Résultat** : Consultez la transcription, la réponse texte et audio
            
            ---
            
            **💡 Astuces :**
            - Parlez distinctement pour une meilleure transcription
            - Posez des questions claires et précises
            - Vérifiez que votre microphone fonctionne
            - Le serveur backend doit être actif sur `http://127.0.0.1:8000`
            """
        )

    # Footer
    with gr.Row(elem_classes="footer"):
        gr.Markdown(
            """
            **⚡ Propulsé par FastAPI + Gradio | 🚗 Renault Group**
            
            _Interface moderne avec IA vocale avancée_
            """
        )

    # Événement de clic
    button.click(
        fn=send_audio,
        inputs=audio_input,
        outputs=[transcription_out, llm_out, tts_audio_out, status_box],
    )

if __name__ == "__main__":

    # Chargement robuste du CSS
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css=css_content,
    )
