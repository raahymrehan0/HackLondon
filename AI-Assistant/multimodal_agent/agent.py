import os
import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, cartesia, deepgram, silero, turn_detector

# Import LlamaIndex libraries
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

# --- Read UCL knowledge from "IOE_info.txt" ---
IOE_FILE_PATH = "IOE_info.txt"
try:
    with open(IOE_FILE_PATH, "r", encoding="utf-8") as file:
        ucl_text = file.read()
    logger.info("Successfully loaded IOE_info.txt")
except FileNotFoundError:
    logger.error(f"File {IOE_FILE_PATH} not found! Using fallback text.")
    ucl_text = "UCL was founded in 1826."

# --- Create or load the UCL knowledge bank index ---
PERSIST_DIR = "./ucl-index-storage"
if not os.path.exists(PERSIST_DIR):
    # Create a document from the file content
    document = Document(text=ucl_text)
    index = VectorStoreIndex.from_documents([document])
    # Persist the index to disk
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# --- Define a function context to query the UCL knowledge bank ---
class UCLKnowledgeBank(llm.FunctionContext):
    @llm.ai_callable(description="Get detailed information about UCL from the knowledge bank")
    async def query_info(self, query: str) -> str:
        # Create an async query engine from the index
        query_engine = index.as_query_engine(use_async=True)
        res = await query_engine.aquery(query)
        logger.info("Query result: %s", res)
        return str(res)

# Create an instance of the function context
ucl_fn_ctx = UCLKnowledgeBank()

# --- Prewarm function for voice agent ---
def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()

# --- Main entrypoint for the voice assistant ---
async def entrypoint(ctx: JobContext):
    # Initial chat context for the assistant
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users is voice-based. "
            "You should use short and concise responses. "
            "In addition to regular conversation, you have access to a comprehensive UCL knowledge bank."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Create the voice assistant agent, integrating the UCL function context
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),  # Using a concrete LLM implementation
        tts=cartesia.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
        fnc_ctx=ucl_fn_ctx,  # add the UCL knowledge bank as an available tool
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)
    # Greet the user
    await agent.say("Hey, how can I help you today?", allow_interruptions=False)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
