#!/usr/bin/env python3

import argparse
import socket
import sys
import os
import logging
import numpy as np
import librosa
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Import line_packet for communication protocol
import line_packet

SAMPLING_RATE = 16000

class WhisperClient:
    def __init__(self, host='localhost', port=43007, min_chunk_size=1.0):
        self.host = host
        self.port = port
        self.min_chunk_size = min_chunk_size
        self.sock = None
        
    def connect(self):
        """Connect to the whisper server"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to whisper server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        if self.sock:
            self.sock.close()
            self.sock = None
            logger.info("Disconnected from server")
    
    def load_audio(self, audio_file):
        """Load audio file and convert to the format expected by the server"""
        try:
            # Check if file exists
            if not os.path.isfile(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            logger.info(f"Loading audio file: {audio_file}")
            
            # Load audio using librosa (handles many formats including MP3)
            # This automatically converts to mono and specified sample rate
            audio_data, sr = librosa.load(audio_file, sr=SAMPLING_RATE, mono=True, dtype=np.float32)
            
            logger.info(f"Loaded audio: {len(audio_data)} samples, {len(audio_data)/SAMPLING_RATE:.2f} seconds")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return None
    
    def load_audio_chunk(self, audio_data, beg, end):
        """Extract a chunk of audio data from beg to end seconds"""
        beg_s = int(beg * SAMPLING_RATE)
        end_s = int(end * SAMPLING_RATE)
        return audio_data[beg_s:end_s]
    
    def send_audio_chunk(self, audio_chunk):
        """Send a chunk of audio data to the server in s16le format"""
        try:
            # Convert to int16 format (s16le)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            # Convert numpy array to bytes (s16le format)
            audio_bytes = audio_int16.tobytes()
            
            # Send raw audio bytes
            self.sock.sendall(audio_bytes)
            
            return True
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
            return False
    
    def receive_transcription(self, timeout=0.1):
        """Receive transcription results from the server"""
        try:
            # Set timeout to avoid blocking
            self.sock.settimeout(timeout)
            # Use line_packet to receive results
            lines = line_packet.receive_lines(self.sock)
            return lines
        except socket.timeout:
            return None
        except (ConnectionResetError, BrokenPipeError):
            # Server closed connection
            return None
        except Exception as e:
            logger.debug(f"No transcription received: {e}")
            return None
        finally:
            # Reset to blocking mode
            self.sock.settimeout(None)
    
    def process_audio_file_realtime(self, audio_file):
        """Process audio file in real-time simulation mode (like whisper_online.py)"""
        # Load the complete audio
        audio_data = self.load_audio(audio_file)
        if audio_data is None:
            return False
        
        # Connect to server
        if not self.connect():
            return False
        
        try:
            duration = len(audio_data) / SAMPLING_RATE
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            beg = 0.0
            end = 0.0
            start_time = time.time()
            
            logger.info("Starting real-time audio streaming...")
            
            while True:
                # Wait until real-time catches up (simulating real-time streaming)
                now = time.time() - start_time
                if now < end + self.min_chunk_size:
                    sleep_time = self.min_chunk_size + end - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Update end time
                end = time.time() - start_time
                
                # Extract audio chunk from beg to end
                a = self.load_audio_chunk(audio_data, beg, end)
                beg = end
                
                if len(a) == 0:
                    break
                
                # Send the audio chunk
                if not self.send_audio_chunk(a):
                    logger.error("Failed to send audio chunk")
                    break
                
                logger.debug(f"Sent chunk: {beg:.2f}s to {end:.2f}s ({len(a)} samples)")
                
                # Try to receive transcriptions (non-blocking)
                transcription = self.receive_transcription()
                if transcription:
                    for line in transcription:
                        print(f"Transcription: {line}")
                
                # Check if we've reached the end
                if end >= duration:
                    logger.info("Reached end of audio")
                    break
            
            # Signal end of audio by shutting down the sending side of the socket
            # but keep receiving until server closes connection
            logger.info("Finished sending audio, shutting down send side of socket...")
            self.sock.shutdown(socket.SHUT_WR)
            
            # Wait for final transcriptions - keep polling until server sends everything
            logger.info("Waiting for final transcriptions...")
            
            # Keep receiving until server closes connection or no more data
            while True:
                try:
                    # Use longer timeout for final data
                    transcription = self.receive_transcription(timeout=2.0)
                    if transcription:
                        for line in transcription:
                            print(f"Final transcription: {line}")
                    else:
                        # No data received, check if connection is still open
                        break
                except Exception as e:
                    logger.debug(f"Connection closed by server: {e}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            self.disconnect()
        
        return True

    def process_audio_file_batch(self, audio_file):
        """Process audio file in computational unaware mode (faster batch processing)"""
        # Load the complete audio
        audio_data = self.load_audio(audio_file)
        if audio_data is None:
            return False
        
        # Connect to server
        if not self.connect():
            return False
        
        try:
            duration = len(audio_data) / SAMPLING_RATE
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            beg = 0.0
            
            logger.info("Starting batch audio processing...")
            
            while True:
                end = beg + self.min_chunk_size
                
                # Extract audio chunk
                a = self.load_audio_chunk(audio_data, beg, end)
                
                if len(a) == 0:
                    break
                
                # Send the audio chunk
                if not self.send_audio_chunk(a):
                    logger.error("Failed to send audio chunk")
                    break
                
                logger.info(f"Sent chunk: {beg:.2f}s to {end:.2f}s ({len(a)} samples)")
                
                # Try to receive transcriptions
                transcription = self.receive_transcription()
                if transcription:
                    for line in transcription:
                        print(f"Transcription: {line}")
                
                logger.debug(f"Last processed: {end:.2f}s")
                
                # Check if we've reached the end
                if end >= duration:
                    logger.info("Reached end of audio")
                    break
                
                beg = end
                
                # Adjust final chunk
                if end + self.min_chunk_size > duration:
                    end = duration
            
            # Signal end of audio by shutting down the sending side of the socket
            # but keep receiving until server closes connection
            logger.info("Finished sending audio, shutting down send side of socket...")
            self.sock.shutdown(socket.SHUT_WR)
            
            # Wait for final transcriptions - keep polling until server sends everything
            logger.info("Waiting for final transcriptions...")
            
            # Keep receiving until server closes connection or no more data
            while True:
                try:
                    # Use longer timeout for final data
                    transcription = self.receive_transcription(timeout=2.0)
                    if transcription:
                        for line in transcription:
                            print(f"Final transcription: {line}")
                    else:
                        # No data received, check if connection is still open
                        break
                except Exception as e:
                    logger.debug(f"Connection closed by server: {e}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            self.disconnect()
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Send audio file to Whisper streaming server")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=43007, help="Server port (default: 43007)")
    parser.add_argument("--min-chunk-size", type=float, default=1.0, 
                       help="Minimum audio chunk size in seconds (default: 1.0)")
    parser.add_argument("--mode", choices=["realtime", "batch"], default="batch",
                       help="Processing mode: realtime (simulates real-time) or batch (faster)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create client
    client = WhisperClient(host=args.host, port=args.port, min_chunk_size=args.min_chunk_size)
    
    # Process the file
    if args.mode == "realtime":
        success = client.process_audio_file_realtime(args.audio_file)
    else:
        success = client.process_audio_file_batch(args.audio_file)
    
    if not success:
        sys.exit(1)
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
