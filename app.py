import os
import asyncio
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from google import genai
from google.genai.types import GenerateContentConfig

from fastmcp import FastMCP, Client

import json


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "deploy_config.json"
INDICES_DIR = BASE_DIR / "indices"


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:
    DEPLOY_CONFIG = json.load(f)


MODEL_CHAT = os.getenv(
    "MODEL_CHAT",
    DEPLOY_CONFIG[
        "model_chat"
    ]
)

MODEL_ROUTING = os.getenv(
    "MODEL_ROUTING",
    DEPLOY_CONFIG[
        "model_routing"
    ]
)

K_GENERACION = int(
    DEPLOY_CONFIG[
        "k_generacion"
    ]
)

UMBRAL_DISTANCIA = float(
    DEPLOY_CONFIG[
        "umbral_distancia"
    ]
)


api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "Falta la variable de entorno GEMINI_API_KEY."
    )


gemini_client = genai.Client(
    api_key=api_key
)


class SentenceEmbeddings(Embeddings):
    def __init__(
        self,
        model_name,
        batch_size=8
    ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_name = model_name
        self.batch_size = batch_size

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                model_name
            )
            .to(
                self.device
            )
            .eval()
        )

    def _encode(
        self,
        textos
    ):
        vectores = []

        for i in range(
            0,
            len(textos),
            self.batch_size
        ):
            lote = textos[
                i:i + self.batch_size
            ]

            entradas = self.tokenizer(
                lote,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(
                self.device
            )

            with torch.no_grad():
                salida = self.model(
                    **entradas
                )

            emb = F.normalize(
                salida.last_hidden_state[
                    :,
                    0
                ],
                p=2,
                dim=1
            )

            vectores.extend(
                emb.cpu().tolist()
            )

        return vectores

    def embed_documents(
        self,
        texts
    ):
        return self._encode(
            list(
                texts
            )
        )

    def embed_query(
        self,
        text
    ):
        return self._encode(
            [
                text
            ]
        )[0]


sentence_embeddings = (
    SentenceEmbeddings(
        DEPLOY_CONFIG.get(
            "embedding_model",
            "BAAI/bge-m3"
        )
    )
)


INDICE_POR_NOMBRE = {}

for nombre in (
    "concepto",
    "codigo",
    "tarea",
    "todos"
):
    ruta = (
        INDICES_DIR
        / nombre
    )

    INDICE_POR_NOMBRE[
        nombre
    ] = FAISS.load_local(
        str(
            ruta
        ),
        sentence_embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=(
            DistanceStrategy.COSINE
        )
    )


MAPA_INDICE = {
    frozenset({
        "slide",
        "transcripcion"
    }): "concepto",

    frozenset({
        "codigo"
    }): "codigo",

    frozenset({
        "slide",
        "transcripcion",
        "codigo",
        "administrativo"
    }): "tarea",
}


def obtener_indice(
    tipos=None
):
    if not tipos:
        return INDICE_POR_NOMBRE[
            "todos"
        ]

    clave = frozenset(
        tipos
    )

    if clave not in MAPA_INDICE:
        raise ValueError(
            "Tipos de contenido no configurados."
        )

    return INDICE_POR_NOMBRE[
        MAPA_INDICE[
            clave
        ]
    ]


def recuperar(
    consulta,
    tipos=None,
    k=K_GENERACION
):
    return obtener_indice(
        tipos
    ).similarity_search_with_score(
        consulta,
        k=k
    )


def formatear_contexto(resultados):
    partes = []

    for doc, distancia in resultados:
        meta = doc.metadata

        origen = (
            f"Ciclo {meta.get('ciclo')} · "
            f"{meta.get('curso', 'Curso no identificado')} · "
            f"{meta.get('fuente', 'Material académico')}"
        )

        if meta.get("clase") is not None:
            origen += f" · clase {meta['clase']}"

        partes.append(
            f"[Fuente: {origen} | distancia {distancia:.3f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(partes)



def _generar(
    system_prompt,
    user_prompt,
    temperature=0.2,
    model=None
):
    modelo = (
        model
        or MODEL_CHAT
    )

    config_kwargs = {
        "system_instruction":
            system_prompt
    }

    if not modelo.startswith(
        "gemini-3"
    ):
        config_kwargs[
            "temperature"
        ] = temperature

    config = GenerateContentConfig(
        **config_kwargs
    )

    mensajes = [
        genai.types.Content(
            role="user",
            parts=[
                genai.types.Part.from_text(
                    text=user_prompt
                )
            ]
        )
    ]

    return (
        gemini_client.models
        .generate_content(
            model=modelo,
            contents=mensajes,
            config=config
        )
        .text
    )


ETIQUETA_FALLBACK = (
    "[Fallback]"
)


def fn_fallback(
    consulta,
    motivo="fuera_alcance"
):
    if motivo == "fuera_alcance":
        return (
            f"{ETIQUETA_FALLBACK} "
            "No encontré evidencia suficiente en el material "
            "del curso para responder esa consulta. "
            "Puedo ayudarte con conceptos, código, notebooks "
            "y entregables del curso."
        )

    if motivo == "sin_codigo":
        return (
            f"{ETIQUETA_FALLBACK} "
            "No encontré fragmentos de código relacionados. "
            "Indícame el notebook, función o fragmento."
        )

    if motivo == "sin_guia":
        return (
            f"{ETIQUETA_FALLBACK} "
            "No encontré material suficiente para construir "
            "una guía sobre esa tarea."
        )

    if motivo == "mensaje_vacio":
        return (
            f"{ETIQUETA_FALLBACK} "
            "Escribe una consulta para poder ayudarte."
        )

    return (
        f"{ETIQUETA_FALLBACK} "
        "No pude completar la consulta en este momento. "
        "Intenta nuevamente o reformula la pregunta."
    )


def fn_responder_concepto(
    pregunta
):
    resultados = recuperar(
        pregunta,
        tipos={
            "slide",
            "transcripcion"
        },
        k=K_GENERACION
    )

    suficiente = (
        bool(
            resultados
        )
        and resultados[0][1]
        <= UMBRAL_DISTANCIA
    )

    if not suficiente:
        return fn_fallback(
            pregunta,
            "fuera_alcance"
        )

    return _generar(
        (
            "Eres TutorBot, asistente del curso de Diseño "
            "de Chatbots Conversacionales. Responde en español "
            "usando únicamente la documentación proporcionada. "
            "No agregues información no sustentada."
        ),
        (
            f"Pregunta: {pregunta}\n\n"
            f"Documentación:\n"
            f"{formatear_contexto(resultados)}"
        )
    )


def fn_explicar_codigo(
    pregunta
):
    resultados = recuperar(
        pregunta,
        tipos={"codigo"},
        k=K_GENERACION
    )

    if not resultados:
        return fn_fallback(
            pregunta,
            "sin_codigo"
        )

    return _generar(
        (
            "Eres TutorBot. Explica el código del material académico "
            "en español usando únicamente los fragmentos "
            "proporcionados. Indica el notebook de origen."
        ),
        (
            f"Consulta: {pregunta}\n\n"
            f"Fragmentos:\n"
            f"{formatear_contexto(resultados)}"
        )
    )


def fn_guiar_tarea(
    tarea,
    detalle=""
):
    if not detalle.strip():
        return (
            f"Para guiarte con '{tarea}' necesito un dato más: "
            "¿sobre qué clase, notebook o entregable "
            "específico lo necesitas?"
        )

    consulta = (
        f"{tarea} {detalle}"
    )

    resultados = recuperar(
        consulta,
        tipos={
            "slide",
            "transcripcion",
            "codigo",
            "administrativo"
        },
        k=K_GENERACION
    )

    if not resultados:
        return fn_fallback(
            consulta,
            "sin_guia"
        )

    return _generar(
        (
            "Eres TutorBot. Entrega una guía numerada "
            "paso a paso, en español, basada únicamente "
            "en el material provisto. Máximo 6 pasos."
        ),
        (
            f"Tarea: {tarea}\n"
            f"Contexto: {detalle}\n\n"
            f"Material:\n"
            f"{formatear_contexto(resultados)}"
        )
    )


def fn_pedir_aclaracion(
    consulta
):
    return _generar(
        (
            "Eres TutorBot. La consulta del estudiante es "
            "ambigua. Devuelve una sola pregunta corta "
            "en español para desambiguarla."
        ),
        f"Consulta ambigua: {consulta}",
        temperature=0.3
    )


mcp_server = FastMCP(
    "tutorbot"
)


@mcp_server.tool()
def responder_concepto(
    pregunta: str
) -> str:
    """Responde teoría, conceptos y preguntas claras de conocimiento."""
    return fn_responder_concepto(
        pregunta
    )


@mcp_server.tool()
def explicar_codigo(
    pregunta: str
) -> str:
    """Explica código existente en los notebooks del curso."""
    return fn_explicar_codigo(
        pregunta
    )


@mcp_server.tool()
def guiar_tarea(
    tarea: str,
    detalle: str = ""
) -> str:
    """Guía tareas de creación, modificación, implementación o despliegue."""
    return fn_guiar_tarea(
        tarea,
        detalle
    )


@mcp_server.tool()
def pedir_aclaracion(
    consulta: str
) -> str:
    """Pide aclaración cuando la consulta es ambigua o incompleta."""
    return fn_pedir_aclaracion(
        consulta
    )


mcp_client = Client(
    mcp_server
)


SYSTEM_AGENTE = """
Eres TutorBot, asistente académico del diplomado de Desarrollo
de Aplicaciones con IA.

Selecciona siempre una herramienta.

MEMORIA DE CONVERSACIÓN:

- Antes de seleccionar una herramienta, revisa el historial reciente.
- Resuelve referencias como "eso", "ese", "esa", "cada uno",
  "el anterior", "el modelo", "la función", "cómo lo hago",
  "dame ejemplos" o expresiones similares usando el historial.
- Si la consulta puede entenderse usando el historial,
  NO uses pedir_aclaracion.
- Al invocar una herramienta, convierte la consulta en una
  pregunta autosuficiente cuando sea necesario.

Ejemplo:

Historial:
Usuario: ¿Qué diferencia hay entre aprendizaje supervisado
y no supervisado?
TutorBot: ...

Nueva consulta:
Dame ejemplos de cada uno

La herramienta debe recibir algo equivalente a:
"Dame ejemplos de aprendizaje supervisado y aprendizaje no supervisado."

Reglas de selección:

1. responder_concepto
   - teoría, definiciones y conceptos;
   - preguntas claras de conocimiento;
   - preguntas de seguimiento sobre conceptos mencionados
     anteriormente en la conversación;
   - el control de alcance se realiza dentro de la herramienta.

2. explicar_codigo
   - funciones, clases, librerías, celdas o fragmentos
     existentes en los notebooks;
   - preguntas de seguimiento sobre código mencionado
     anteriormente;
   - una pregunta que empieza por "cómo" sigue siendo de código
     si pregunta cómo funciona, se define, se configura o se usa
     código existente.

3. guiar_tarea
   - el estudiante expresa intención de crear, modificar,
     implementar, desplegar o completar algo;
   - preguntas sobre requisitos de una actividad o entregable;
   - preguntas de seguimiento sobre una tarea previamente
     mencionada.

4. pedir_aclaracion
   - úsala solamente si la consulta sigue siendo ambigua
     después de revisar el historial;
   - no la uses si el referente puede deducirse de mensajes
     anteriores;
   - no la uses para una pregunta clara solo porque esté
     fuera del alcance del corpus.

El fallback se activa automáticamente cuando no existe evidencia
suficiente en el corpus o cuando la interacción no puede completarse.

Devuelve la respuesta de la herramienta en español.
Si contiene la etiqueta [Fallback], consérvala.
"""


def construir_mensajes(
    consulta,
    historial=None,
    max_mensajes=8
):
    mensajes = []

    historial = (
        historial
        or []
    )[-max_mensajes:]

    for turno in historial:
        role = turno.get(
            "role",
            "user"
        )

        contenido = turno.get(
            "content",
            ""
        )

        if role == "assistant":
            role = "model"

        if role not in {
            "user",
            "model"
        }:
            continue

        if not isinstance(
            contenido,
            str
        ):
            continue

        if not contenido.strip():
            continue

        mensajes.append(
            genai.types.Content(
                role=role,
                parts=[
                    genai.types.Part.from_text(
                        text=contenido
                    )
                ]
            )
        )

    mensajes.append(
        genai.types.Content(
            role="user",
            parts=[
                genai.types.Part.from_text(
                    text=consulta
                )
            ]
        )
    )

    return mensajes


async def agente(
    consulta,
    historial=None,
    max_iterations=3
):
    consulta = (
        consulta
        or ""
    ).strip()

    if not consulta:
        return (
            fn_fallback(
                "",
                "mensaje_vacio"
            ),
            ["fallback"]
        )

    for intento in range(
        1,
        max_iterations + 1
    ):
        try:
            async with mcp_client:
                config = GenerateContentConfig(
                    system_instruction=(
                        SYSTEM_AGENTE
                    ),
                    tools=[
                        mcp_client.session
                    ],
                )

                respuesta = (
                    await gemini_client.aio.models
                    .generate_content(
                        model=MODEL_ROUTING,
                        contents=construir_mensajes(
                            consulta,
                            historial
                        ),
                        config=config
                    )
                )

            usadas = []

            for contenido in (
                respuesta.automatic_function_calling_history
                or []
            ):
                for parte in (
                    contenido.parts
                    or []
                ):
                    if getattr(
                        parte,
                        "function_call",
                        None
                    ):
                        usadas.append(
                            parte.function_call.name
                        )

            texto = (
                respuesta.text
                or ""
            ).strip()

            if not texto:
                raise ValueError(
                    "respuesta vacía"
                )

            return (
                texto,
                usadas
            )

        except Exception as e:
            if intento == max_iterations:
                return (
                    fn_fallback(
                        consulta,
                        "error_temporal"
                    ),
                    ["fallback"]
                )

            mensaje = str(
                e
            )

            if (
                "429" in mensaje
                or "RESOURCE_EXHAUSTED" in mensaje
                or "503" in mensaje
                or "UNAVAILABLE" in mensaje
            ):
                espera = min(
                    10 * intento,
                    60
                )
            else:
                espera = (
                    1.5 * intento
                )

            await asyncio.sleep(
                espera
            )


async def responder_gradio(
    message,
    history
):
    respuesta, _ = await agente(
        message,
        historial=history
    )

    return respuesta


demo = gr.ChatInterface(
    fn=responder_gradio,
    title="TutorBot",
    description=(
        "Asistente académico del curso de "
        "Diseño de Chatbots Conversacionales"
    ),
    examples=[
        "¿Qué diferencia hay entre aprendizaje supervisado y no supervisado?",
        "¿Qué hace la clase QLearningAgent?",
        "Quiero crear un dashboard con Streamlit",
        "¿Qué es una CNN?",
        "Quiero implementar RAG en mi chatbot",
        "¿Qué caracteriza a una serie temporal?",
    ],
    save_history=True,
)


if __name__ == "__main__":
    port = int(
        os.getenv(
            "PORT",
            "7860"
        )
    )

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
