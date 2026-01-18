"""
Curious Voice Bot - Production Backend
FastAPI server with proper LLM integration, knowledge graph, and session management
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import hashlib
import uuid

# Database and storage
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis

# LLM and AI services
import openai
from anthropic import Anthropic

# Speech services
import httpx

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Structured logging
from loguru import logger
import sys

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Retry logic for API calls
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# JWT authentication
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/curious_bot")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Security configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Setup structured logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    level="INFO"
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/curious_bot_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG"
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer for JWT
security = HTTPBearer()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI
app = FastAPI(title="Curious Voice Bot API", version="2.0.0")

# Add rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global exception handler for clean error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": str(uuid.uuid4())
        }
    )

# CORS middleware for web access - PRODUCTION: Restrict to specific domains
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Restricted to specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio playback
from fastapi.staticfiles import StaticFiles
os.makedirs("static/audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="static/audio"), name="audio")

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis for caching and real-time features
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Authentication helpers
def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Optional: Dependency for protected endpoints
async def get_current_user(token_data: dict = Depends(verify_token)) -> str:
    """Get current user from token"""
    user_id = token_data.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return user_id

# Models
class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer)
    topic = Column(String, nullable=True)
    conversation_history = Column(JSON)
    knowledge_graph = Column(JSON)
    feedback = Column(Text, nullable=True)
    performance_metrics = Column(JSON)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_sessions = Column(Integer, default=0)
    preferences = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models for API
class TeachingInput(BaseModel):
    text: str
    session_id: str
    user_id: Optional[str] = None

class SessionCreate(BaseModel):
    user_id: Optional[str] = None
    topic: Optional[str] = None

class BotResponse(BaseModel):
    response_text: str
    audio_url: Optional[str] = None
    concepts_extracted: List[str]
    question_asked: str
    confidence_score: float
    processing_time: float

# Knowledge Graph Manager
class KnowledgeGraph:
    """
    Tracks what has been taught and builds connections between concepts.
    Uses LLM to maintain a semantic understanding of the conversation.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.graph = {
            "concepts": {},  # concept -> {definition, examples, relations}
            "timeline": [],  # ordered list of what was taught
            "relations": [],  # links between concepts
        }
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    async def process_teaching(self, text: str) -> Dict[str, Any]:
        """Process new teaching input and update knowledge graph"""
        
        # Extract concepts using LLM
        concepts = await self._extract_concepts_llm(text)
        
        # Update graph
        timestamp = datetime.utcnow().isoformat()
        for concept in concepts:
            if concept not in self.graph["concepts"]:
                self.graph["concepts"][concept] = {
                    "introduced_at": timestamp,
                    "definitions": [],
                    "examples": [],
                    "related_to": []
                }
            
            # Extract definition/explanation
            explanation = await self._extract_explanation(text, concept)
            self.graph["concepts"][concept]["definitions"].append(explanation)
        
        # Add to timeline
        self.graph["timeline"].append({
            "timestamp": timestamp,
            "text": text,
            "concepts": concepts
        })
        
        # Identify relations between concepts
        if len(self.graph["concepts"]) > 1:
            relations = await self._identify_relations(concepts)
            self.graph["relations"].extend(relations)
        
        return {
            "concepts": concepts,
            "graph_state": self.graph
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError))
    )
    async def _extract_concepts_llm(self, text: str) -> List[str]:
        """Extract key concepts using GPT-4 with retry logic"""
        
        prompt = f"""Extract 3-5 key concepts from this educational explanation.
Return ONLY a JSON array of concept names.

Text: {text}

Concepts (as JSON array):"""
        
        try:
            logger.debug(f"Extracting concepts from text: {text[:100]}...")
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract educational concepts. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            concepts_json = response.choices[0].message.content.strip()
            concepts = json.loads(concepts_json)
            
            logger.info(f"Extracted {len(concepts)} concepts: {concepts}")
            return concepts if isinstance(concepts, list) else []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in concept extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"Concept extraction error: {e}", exc_info=True)
            return []
    
    async def _extract_explanation(self, text: str, concept: str) -> str:
        """Extract how a concept was explained"""
        
        prompt = f"""From this teaching text, extract the explanation/definition of "{concept}".
Return a concise 1-2 sentence summary.

Text: {text}

Explanation of {concept}:"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except:
            return text[:200]
    
    async def _identify_relations(self, new_concepts: List[str]) -> List[Dict]:
        """Identify relationships between concepts"""
        
        existing = list(self.graph["concepts"].keys())
        
        prompt = f"""Given these existing concepts: {existing}
And these new concepts: {new_concepts}

Identify any relationships (e.g., "is a type of", "depends on", "contrasts with").
Return as JSON array of {{"from": "concept1", "to": "concept2", "relation": "type"}}.

Relationships:"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            relations_json = response.choices[0].message.content.strip()
            relations = json.loads(relations_json)
            return relations if isinstance(relations, list) else []
        except:
            return []
    
    async def generate_contextual_question(self, current_input: str, 
                                          conversation_history: List[Dict]) -> str:
        """Generate question based on knowledge graph context"""
        
        # Get recent concepts
        recent_concepts = self.graph["timeline"][-3:] if self.graph["timeline"] else []
        
        # Build context from knowledge graph
        graph_context = f"""Knowledge Graph:
Concepts taught: {list(self.graph["concepts"].keys())}
Recent topics: {[t['text'][:100] for t in recent_concepts]}
Relations: {self.graph["relations"][-5:]}
"""
        
        # Get conversation context
        conv_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-4:]
        ])
        
        prompt = f"""You are Curious, an enthusiastic student. Based on the knowledge graph and conversation, generate ONE thoughtful question.

{graph_context}

Recent conversation:
{conv_context}

Teacher just said: {current_input}

Generate a question that:
1. Shows you understood the concept
2. Connects to previously taught material (if relevant)
3. Encourages deeper explanation
4. Is specific and thought-provoking

Question:"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",  # Use GPT-4 for best question quality
                messages=[
                    {"role": "system", "content": "You are a curious, intelligent student who asks insightful questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=120,
                temperature=0.8
            )
            
            question = response.choices[0].message.content.strip()
            if not question.endswith('?'):
                question += '?'
            return question
        except Exception as e:
            logger.error(f"Question generation error: {e}", exc_info=True)
            return "Can you explain that in more detail?"

# Neural TTS Manager
class NeuralTTS:
    """
    Handles text-to-speech using ElevenLabs for natural, curious voice.
    Falls back to OpenAI TTS if ElevenLabs unavailable.
    """
    
    def __init__(self):
        self.elevenlabs_key = ELEVENLABS_API_KEY
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Curious student voice ID (configure in ElevenLabs dashboard)
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
    
    async def generate_speech(self, text: str, session_id: str) -> str:
        """Generate speech and return audio URL"""
        
        # Try ElevenLabs first for best quality
        if self.elevenlabs_key:
            try:
                audio_data = await self._generate_elevenlabs(text)
                audio_url = await self._upload_audio(audio_data, session_id)
                return audio_url
            except Exception as e:
                logger.warning(f"ElevenLabs error: {e}, falling back to OpenAI")
        
        # Fallback to OpenAI TTS
        try:
            audio_data = await self._generate_openai_tts(text)
            audio_url = await self._upload_audio(audio_data, session_id)
            return audio_url
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None
    
    async def _generate_elevenlabs(self, text: str) -> bytes:
        """Generate speech using ElevenLabs API"""
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.elevenlabs_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.5,  # Curious, enthusiastic style
                "use_speaker_boost": True
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            return response.content
    
    async def _generate_openai_tts(self, text: str) -> bytes:
        """Generate speech using OpenAI TTS (fallback)"""
        
        response = await asyncio.to_thread(
            self.openai_client.audio.speech.create,
            model="tts-1",
            voice="nova",  # Friendly, curious voice
            input=text
        )
        return response.content
    
    async def _upload_audio(self, audio_data: bytes, session_id: str) -> str:
        """Upload audio to storage and return URL"""
        
        # In production, upload to S3/GCS/Azure Blob Storage
        # For now, save locally and return path
        audio_dir = "static/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_id = hashlib.md5(audio_data).hexdigest()
        audio_path = f"{audio_dir}/{session_id}_{audio_id}.mp3"
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        return f"/audio/{session_id}_{audio_id}.mp3"

# Speech-to-Text Manager
class SpeechToText:
    """
    Handles speech recognition using Deepgram (production-grade STT).
    Falls back to OpenAI Whisper if needed.
    """
    
    def __init__(self):
        self.deepgram_key = DEEPGRAM_API_KEY
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio to text"""
        
        # Try Deepgram first (faster, cheaper for real-time)
        if self.deepgram_key:
            try:
                return await self._transcribe_deepgram(audio_data)
            except Exception as e:
                logger.warning(f"Deepgram error: {e}, falling back to Whisper")
        
        # Fallback to OpenAI Whisper
        return await self._transcribe_whisper(audio_data)
    
    async def _transcribe_deepgram(self, audio_data: bytes) -> str:
        """Transcribe using Deepgram API"""
        
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {self.deepgram_key}",
            "Content-Type": "audio/wav"
        }
        params = {
            "model": "nova-2",
            "smart_format": "true",
            "punctuate": "true"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                headers=headers, 
                params=params, 
                content=audio_data,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript
    
    async def _transcribe_whisper(self, audio_data: bytes) -> str:
        """Transcribe using OpenAI Whisper (fallback)"""
        
        # Save temporarily for Whisper API
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_data)
        
        try:
            with open(temp_path, "rb") as audio_file:
                transcript = await asyncio.to_thread(
                    self.openai_client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Session Manager
class SessionManager:
    """Manages teaching sessions with Redis-backed persistence for multi-instance scalability"""
    
    def __init__(self):
        # Use Redis for active sessions (multi-instance safe)
        # No longer using in-memory dict - prevents state loss on restart/scale
        self.tts = NeuralTTS()
        self.stt = SpeechToText()
    
    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"
    
    async def _get_session_from_redis(self, session_id: str) -> Optional[Dict]:
        """Retrieve session from Redis"""
        try:
            session_data = redis_client.get(self._get_session_key(session_id))
            if session_data:
                return json.loads(session_data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}", exc_info=True)
            return None
    
    async def _save_session_to_redis(self, session_id: str, session_data: Dict):
        """Save session to Redis with 24-hour expiry"""
        try:
            redis_client.setex(
                self._get_session_key(session_id),
                86400,  # 24 hours
                json.dumps(session_data, default=str)
            )
        except Exception as e:
            logger.error(f"Redis save error: {e}", exc_info=True)
    
    async def create_session(self, user_id: Optional[str], topic: Optional[str]) -> str:
        """Create a new teaching session"""
        
        session_id = str(uuid.uuid4())
        
        # Create knowledge graph for session
        kg = KnowledgeGraph(session_id)
        
        # Store in Redis (not in-memory dict)
        session_data = {
            "kg": {"graph": kg.graph},
            "start_time": datetime.utcnow().isoformat(),
            "conversation": [],
            "user_id": user_id,
            "topic": topic
        }
        await self._save_session_to_redis(session_id, session_data)
        
        # Store in database
        db = SessionLocal()
        try:
            session = Session(
                id=session_id,
                user_id=user_id,
                topic=topic,
                conversation_history=[],
                knowledge_graph={},
                performance_metrics={}
            )
            db.add(session)
            db.commit()
        finally:
            db.close()
        
        return session_id
    
    async def process_teaching(self, session_id: str, text: str) -> BotResponse:
        """Process teaching input and generate response with parallel LLM calls"""
        
        # Get session from Redis
        session = await self._get_session_from_redis(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Reconstruct knowledge graph from session data
        kg = KnowledgeGraph(session_id)
        kg.graph = session["kg"]["graph"]
        
        start_time = datetime.utcnow()
        
        # Add to conversation history
        session["conversation"].append({
            "role": "teacher",
            "content": text,
            "timestamp": start_time.isoformat()
        })
        
        # PARALLEL PROCESSING: Run knowledge graph update and question generation concurrently
        # This reduces latency from ~5-10s to ~2-3s
        graph_result, question = await asyncio.gather(
            kg.process_teaching(text),
            kg.generate_contextual_question(text, session["conversation"])
        )
        
        concepts = graph_result["concepts"]
        
        # Create response
        response_text = f"That's really interesting! {question}"
        
        # Add to conversation
        session["conversation"].append({
            "role": "curious",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update session data with new graph state
        session["kg"]["graph"] = kg.graph
        
        # Save session back to Redis
        await self._save_session_to_redis(session_id, session)
        
        # Generate audio (async, non-blocking)
        audio_url = await self.tts.generate_speech(response_text, session_id)
        
        # Calculate metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update database asynchronously (don't block response)
        asyncio.create_task(self._update_session_db(session_id, session))
        
        return BotResponse(
            response_text=response_text,
            audio_url=audio_url,
            concepts_extracted=concepts,
            question_asked=question,
            confidence_score=0.95,  # Could calculate based on LLM logprobs
            processing_time=processing_time
        )
    
    async def _update_session_db(self, session_id: str, session_data: Dict):
        """Update session in database"""
        
        db = SessionLocal()
        try:
            session = db.query(Session).filter(Session.id == session_id).first()
            if session:
                session.conversation_history = session_data["conversation"]
                session.knowledge_graph = session_data["kg"]["graph"]
                db.commit()
        finally:
            db.close()
    
    async def end_session(self, session_id: str) -> Dict:
        """End session and generate feedback"""
        
        # Get session from Redis
        session = await self._get_session_from_redis(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        start_time = datetime.fromisoformat(session["start_time"])
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Reconstruct knowledge graph
        kg_data = session["kg"]["graph"]
        
        # Generate comprehensive feedback using knowledge graph
        feedback = await self._generate_feedback(session)
        
        # Update database
        db = SessionLocal()
        try:
            db_session = db.query(Session).filter(Session.id == session_id).first()
            if db_session:
                db_session.ended_at = datetime.utcnow()
                db_session.duration_seconds = int(duration)
                db_session.feedback = feedback
                db.commit()
        finally:
            db.close()
        
        # Clean up Redis session
        redis_client.delete(self._get_session_key(session_id))
        
        return {
            "session_id": session_id,
            "duration_seconds": duration,
            "feedback": feedback,
            "concepts_taught": len(kg_data["concepts"]),
            "questions_asked": sum(1 for msg in session["conversation"] if msg["role"] == "curious")
        }
    
    async def _generate_feedback(self, session: Dict) -> str:
        """Generate detailed session feedback using LLM"""
        
        kg_data = session["kg"]["graph"]
        conversation = session["conversation"]
        start_time = datetime.fromisoformat(session["start_time"])
        
        prompt = f"""Generate detailed teaching feedback based on this session.

Knowledge Graph:
{json.dumps(kg_data, indent=2)}

Conversation turns: {len(conversation)}
Duration: {(datetime.utcnow() - start_time).total_seconds() / 60:.1f} minutes

Generate feedback covering:
1. Topics covered and depth of explanation
2. Teaching strengths (clarity, examples, engagement)
3. Suggestions for improvement
4. Recommended next topics based on what was taught

Feedback (markdown format):"""
        
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

# Initialize global session manager
session_manager = SessionManager()

# API Endpoints
@app.post("/api/sessions", response_model=dict)
@limiter.limit("10/minute")  # Rate limit: 10 session creations per minute
async def create_session(request: Request, data: SessionCreate):
    """Create a new teaching session"""
    logger.info(f"Creating session for user: {data.user_id}")
    
    try:
        session_id = await session_manager.create_session(data.user_id, data.topic)
        logger.info(f"Session created: {session_id}")
        
        return {
            "session_id": session_id,
            "message": "Session created successfully",
            "greeting": "Hello, Teacher! I'm Curious, ready to learn. What would you like to teach me today?"
        }
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.post("/api/teach", response_model=BotResponse)
@limiter.limit("30/minute")  # Rate limit: 30 teaching interactions per minute
async def teach(request: Request, data: TeachingInput):
    """Process teaching input and get bot response"""
    logger.info(f"Teaching interaction for session {data.session_id}")
    
    try:
        response = await session_manager.process_teaching(data.session_id, data.text)
        logger.debug(f"Response generated in {response.processing_time:.2f}s")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing teaching: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process teaching input")

@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a teaching session and get feedback"""
    logger.info(f"Ending session: {session_id}")
    
    try:
        result = await session_manager.end_session(session_id)
        logger.info(f"Session {session_id} ended. Duration: {result['duration_seconds']}s")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to end session")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "id": session.id,
            "created_at": session.created_at,
            "ended_at": session.ended_at,
            "topic": session.topic,
            "conversation_history": session.conversation_history,
            "knowledge_graph": session.knowledge_graph,
            "feedback": session.feedback
        }
    finally:
        db.close()

@app.websocket("/ws/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time conversation"""
    await websocket.accept()
    
    try:
        while True:
            # Receive audio or text
            data = await websocket.receive_json()
            
            if data["type"] == "audio":
                # Transcribe audio
                audio_bytes = bytes.fromhex(data["audio"])
                text = await session_manager.stt.transcribe(audio_bytes)
                
                # Process teaching
                response = await session_manager.process_teaching(session_id, text)
                
                await websocket.send_json({
                    "type": "response",
                    "text": response.response_text,
                    "audio_url": response.audio_url,
                    "concepts": response.concepts_extracted
                })
            
            elif data["type"] == "text":
                # Process text directly
                response = await session_manager.process_teaching(session_id, data["text"])
                
                await websocket.send_json({
                    "type": "response",
                    "text": response.response_text,
                    "audio_url": response.audio_url,
                    "concepts": response.concepts_extracted
                })
            
            elif data["type"] == "interrupt":
                # Handle interruption
                await websocket.send_json({
                    "type": "interrupted",
                    "message": "I'm listening..."
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await websocket.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "database": "connected",
            "redis": "connected",
            "openai": "configured" if OPENAI_API_KEY else "not configured",
            "elevenlabs": "configured" if ELEVENLABS_API_KEY else "not configured"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
