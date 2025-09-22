#!/usr/bin/env python3

import numpy as np
import torch
import time
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque

logger = logging.getLogger(__name__)

class StreamingDiarizer:
    """
    Streaming speaker diarization optimized for integration with whisper server
    """
    
    def __init__(self, 
                 window_size: float = 1.5,           # seconds per embedding window
                 step_size: float = 0.75,            # hop size between windows
                 sample_rate: int = 16000,
                 similarity_threshold: float = 0.75,  # speaker similarity threshold
                 vad_threshold: float = 0.5,         # voice activity threshold
                 min_speech_duration: float = 0.5):  # minimum speech duration to process
        
        self.window_size = window_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        
        # Calculate samples
        self.num_samples_window = int(window_size * sample_rate)
        self.num_samples_step = int(step_size * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = deque()
        self.buffer_duration = 0.0
        self.total_processed_time = 0.0
        
        # Speaker tracking
        self.speaker_profiles: List[np.ndarray] = []
        self.speaker_names = []
        self.current_speakers = {}  # timestamp -> speaker_id mapping
        
        # Initialize models
        self.models_loaded = False
        self._try_load_models()
        
        logger.info(f"StreamingDiarizer initialized (models_loaded: {self.models_loaded})")
    
    def _try_load_models(self):
        """Try to load diarization models, continue without them if they fail"""
        try:
            # Try to import and load models
            from resemblyzer import VoiceEncoder, preprocess_wav
            
            # Initialize Silero VAD
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            self.vad_model.eval()
            # Force VAD model to CPU to avoid CUDA conflicts with Whisper
            self.vad_model = self.vad_model.cpu()
            self.get_speech_timestamps = utils[0]
            
            # Initialize Resemblyzer encoder on CPU
            self.encoder = VoiceEncoder(device="cpu")
            self.preprocess_wav = preprocess_wav
            
            self.models_loaded = True
            logger.info("Diarization models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load diarization models: {e}")
            logger.warning("Continuing without diarization - install 'resemblyzer' and 'silero-vad' for speaker identification")
            self.models_loaded = False
    
    def is_enabled(self) -> bool:
        """Check if diarization is available"""
        return self.models_loaded
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer for processing"""
        if not self.models_loaded:
            return
        
        self.audio_buffer.append(audio_chunk)
        self.buffer_duration += len(audio_chunk) / self.sample_rate
    
    def should_process(self) -> bool:
        """Check if we have enough audio to process"""
        return self.models_loaded and self.buffer_duration >= self.window_size
    
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Use Silero VAD to detect speech segments in audio"""
        if not self.models_loaded:
            duration = len(audio) / self.sample_rate
            return [(0.0, duration)]
        
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_speech_duration_ms=int(self.min_speech_duration * 1000),
                min_silence_duration_ms=100,
                window_size_samples=512,
                speech_pad_ms=30
            )
            
            # Convert to seconds
            segments = []
            for segment in speech_timestamps:
                start_sec = segment['start'] / self.sample_rate
                end_sec = segment['end'] / self.sample_rate
                segments.append((start_sec, end_sec))
            
            return segments
            
        except Exception as e:
            logger.warning(f"VAD failed: {e}")
            # Fallback: assume entire audio is speech
            duration = len(audio) / self.sample_rate
            return [(0.0, duration)]
    
    def extract_embedding(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio segment"""
        if not self.models_loaded:
            return None
        
        try:
            # Skip very short segments
            if len(audio_segment) < self.min_speech_samples:
                return None
            
            # Preprocess for Resemblyzer
            processed_audio = self.preprocess_wav(audio_segment, source_sr=self.sample_rate)
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(processed_audio)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None
    
    def identify_speaker(self, embedding: np.ndarray) -> int:
        """Identify speaker based on embedding similarity"""
        if not self.speaker_profiles:
            # First speaker
            self.speaker_profiles.append(embedding.copy())
            speaker_id = 0
            self.speaker_names.append(f"SPEAKER_{speaker_id:02d}")
            logger.debug(f"New speaker detected: {self.speaker_names[speaker_id]}")
            return speaker_id
        
        # Calculate similarities with existing speakers
        similarities = []
        for profile in self.speaker_profiles:
            # Cosine similarity
            similarity = np.dot(embedding, profile) / (
                np.linalg.norm(embedding) * np.linalg.norm(profile)
            )
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        
        if max_similarity >= self.similarity_threshold:
            # Match with existing speaker
            speaker_id = similarities.index(max_similarity)
            
            # Update speaker profile (running average)
            alpha = 0.1  # Learning rate
            self.speaker_profiles[speaker_id] = (
                (1 - alpha) * self.speaker_profiles[speaker_id] + 
                alpha * embedding
            )
            
            return speaker_id
        else:
            # New speaker
            speaker_id = len(self.speaker_profiles)
            self.speaker_profiles.append(embedding.copy())
            self.speaker_names.append(f"SPEAKER_{speaker_id:02d}")
            logger.info(f"New speaker detected: {self.speaker_names[speaker_id]} (similarity: {max_similarity:.3f})")
            return speaker_id
    
    def process_chunk(self) -> Dict[Tuple[float, float], str]:
        """Process accumulated audio and return speaker assignments"""
        if not self.should_process():
            return {}
        
        # Concatenate buffered audio
        audio_data = np.concatenate(list(self.audio_buffer))
        
        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio_data)
        
        # Process each speech segment for speaker identification
        speaker_assignments = {}
        
        for start_time, end_time in speech_segments:
            # Extract audio segment
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if end_sample > len(audio_data):
                end_sample = len(audio_data)
            
            if start_sample >= end_sample:
                continue
            
            audio_segment = audio_data[start_sample:end_sample]
            
            # Skip very short segments
            if len(audio_segment) < self.min_speech_samples:
                continue
            
            # Extract speaker embedding
            embedding = self.extract_embedding(audio_segment)
            
            if embedding is not None:
                # Identify speaker
                speaker_id = self.identify_speaker(embedding)
                
                # Use absolute timestamps from audio start (not relative to processed time)
                # This aligns with Whisper's timestamp reference
                abs_start_time = self.total_processed_time + start_time
                abs_end_time = self.total_processed_time + end_time
                
                speaker_assignments[(abs_start_time, abs_end_time)] = self.speaker_names[speaker_id]
        
        # Keep overlap for next processing
        overlap_samples = int(self.step_size * self.sample_rate)
        if len(audio_data) > overlap_samples:
            overlap_audio = audio_data[-overlap_samples:]
            self.audio_buffer.clear()
            self.audio_buffer.append(overlap_audio)
            self.buffer_duration = self.step_size
            self.total_processed_time += (len(audio_data) - overlap_samples) / self.sample_rate
        else:
            self.audio_buffer.clear()
            self.buffer_duration = 0.0
            self.total_processed_time += len(audio_data) / self.sample_rate
        
        # Update current speaker assignments
        self.current_speakers.update(speaker_assignments)
        
        # Clean up old assignments (keep last 30 seconds)
        cutoff_time = self.total_processed_time - 30.0
        self.current_speakers = {
            k: v for k, v in self.current_speakers.items() 
            if k[1] > cutoff_time  # Keep if end time is recent
        }
        
        return speaker_assignments
    
    def get_speaker_for_timestamp(self, timestamp: float) -> Optional[str]:
        """Get speaker ID for a specific timestamp"""
        for (start, end), speaker_id in self.current_speakers.items():
            if start <= timestamp <= end:
                return speaker_id
        return None
    
    def get_speaker_stats(self) -> Dict:
        """Get current speaker statistics"""
        return {
            'total_speakers': len(self.speaker_profiles),
            'speaker_names': self.speaker_names.copy(),
            'total_processed_time': self.total_processed_time,
            'models_loaded': self.models_loaded
        }