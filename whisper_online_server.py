#!/usr/bin/env python3
from whisper_online import *

import sys
import argparse
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)

# diarization options
parser.add_argument("--enable-diarization", action="store_true", 
                   help="Enable speaker diarization using Resemblyzer and Silero VAD")
parser.add_argument("--diarization-similarity-threshold", type=float, default=0.75,
                   help="Speaker similarity threshold for diarization (default: 0.75)")

args = parser.parse_args()

set_logging(args,logger,other="")

# setting whisper object by args 

SAMPLING_RATE = 16000

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# warm up the ASR because the very first transcribe takes more time than the others. 
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file,0,1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. "+msg)
        sys.exit(1)
else:
    logger.warning(msg)


######### Server objects

import line_packet
import socket

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes # was: 65536
# whis
    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None


import io
import soundfile

# Import diarization if enabled
if args.enable_diarization:
    from streaming_diarizer import StreamingDiarizer

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk, enable_diarization=False, diarization_similarity_threshold=0.75):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None
        self.is_first = True
        
        # Diarization setup
        self.enable_diarization = enable_diarization
        self.diarizer = None
        
        if enable_diarization:
            try:
                logger.info("Attempting to initialize diarization...")
                self.diarizer = StreamingDiarizer(
                    similarity_threshold=diarization_similarity_threshold
                )
                if self.diarizer.is_enabled():
                    logger.info("✓ Diarization enabled and models loaded successfully")
                else:
                    logger.warning("✗ Diarization models not available - continuing without speaker identification")
                    logger.warning("Install dependencies: pip install resemblyzer silero-vad")
                    self.enable_diarization = False
            except Exception as e:
                logger.warning(f"✗ Failed to initialize diarization: {e}")
                logger.warning("Install dependencies: pip install resemblyzer silero-vad")
                self.enable_diarization = False
        else:
            logger.info("Diarization disabled (use --enable-diarization to enable)")

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
#            print("received audio:",len(raw_bytes), "bytes", raw_bytes[:10])
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self,o):
        # output format is like:
        # 0 1720 0.95 Takhle to je [SPEAKER_00] [word1:0.89, word2:0.92, ...]
        # - the first three values are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model
        #    - average probability/confidence score for the segment
        # - the next value: segment transcript
        # - followed by speaker ID if diarization is enabled
        # - the last part: individual word probabilities in brackets

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            # Handle both old 3-tuple and new 5-tuple formats
            if len(o) >= 5:
                beg, end, text, avg_prob, word_probs = o[0]*1000, o[1]*1000, o[2], o[3], o[4]
            else:
                beg, end, text = o[0]*1000, o[1]*1000, o[2]
                avg_prob = 1.0
                word_probs = []
                
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            
            # Add speaker identification if diarization is enabled
            speaker_info = ""
            if self.enable_diarization and self.diarizer:
                # Get speaker for the middle of the segment
                mid_timestamp = (beg + end) / 2000.0  # Convert to seconds
                
                # Use absolute timestamp from start of audio file
                # Whisper timestamps are already absolute from audio start
                speaker_id = self.diarizer.get_speaker_for_timestamp(mid_timestamp)
                
                if speaker_id:
                    speaker_info = f" [{speaker_id}]"
                    logger.debug(f"Speaker identified: {speaker_id} for timestamp {mid_timestamp:.2f}s")
                else:
                    logger.debug(f"No speaker found for timestamp {mid_timestamp:.2f}s")
                    # Debug: show available speaker segments
                    if hasattr(self.diarizer, 'current_speakers') and self.diarizer.current_speakers:
                        segments_info = [(f"{s:.2f}-{e:.2f}s:{spk}" ) for (s,e), spk in self.diarizer.current_speakers.items()]
                        logger.debug(f"Available segments: {segments_info}")
            
            # Format word probabilities for display
            word_prob_str = ", ".join([f"{word}:{prob:.3f}" for word, prob in word_probs]) if word_probs else ""
            if word_prob_str:
                word_prob_str = f" [words: {word_prob_str}]"
            
            output_msg = "%1.0f %1.0f %1.3f %s%s%s" % (beg, end, avg_prob, text, speaker_info, word_prob_str)
            print(output_msg, flush=True, file=sys.stderr)
            return output_msg
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            
            # Add audio to diarization buffer
            if self.enable_diarization and self.diarizer:
                self.diarizer.add_audio_chunk(a)
                
                # Process diarization if enough audio is buffered
                if self.diarizer.should_process():
                    speaker_assignments = self.diarizer.process_chunk()
                    if speaker_assignments:
                        logger.debug(f"Diarization processed, found {len(speaker_assignments)} speaker segments")
                        for (start, end), speaker in speaker_assignments.items():
                            logger.debug(f"  {start:.2f}-{end:.2f}s: {speaker}")
            
            # Process ASR as usual
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

        # Process any remaining audio in the buffer
        try:
            o = self.online_asr_proc.finish()
            self.send_result(o)
        except BrokenPipeError:
            logger.info("broken pipe -- connection closed during finish")
        
        # Log final diarization stats
        if self.enable_diarization and self.diarizer:
            stats = self.diarizer.get_speaker_stats()
            if stats['total_speakers'] > 0:
                logger.info(f"Session complete: {stats['total_speakers']} speakers detected: {', '.join(stats['speaker_names'])}")
            else:
                logger.info("Session complete: No speakers detected")



# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((args.host, args.port))
    s.listen(1)
    logger.info('Listening on'+str((args.host, args.port)))
    while True:
        conn, addr = s.accept()
        logger.info('Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(
            connection, 
            online, 
            args.min_chunk_size, 
            enable_diarization=args.enable_diarization,
            diarization_similarity_threshold=args.diarization_similarity_threshold
        )
        proc.process()
        conn.close()
        logger.info('Connection to client closed')
logger.info('Connection closed, terminating.')
