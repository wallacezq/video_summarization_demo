from typing import Iterator, Dict, List, Tuple
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
import cv2
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model for human detection
model = YOLO("yolov8n.pt")
# model.classes = [0] # this limits people as the only class that is detected by yolo 

def run_vlm_inference(chunk_path: str, formatted_time: str):
    """Run VLM inference on a video chunk and generate text description."""
    try:
        # Dummy VLM inference
        print("LVLM Inference done!")
        
        # Uncomment this line to delete processed chunks
        # os.remove(chunk_path)
        print(f"[INFO] Processed chunk {chunk_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process chunk {chunk_path}: {e}")

# Initialize thread pool for parallel inference
executor = ThreadPoolExecutor(max_workers=2)

class RTSPChunkLoader(BaseLoader):
    def __init__(self, rtsp_url: str, chunk_type: str, chunk_args: Dict, output_dir: str = "output_chunks"):
        self.rtsp_url = rtsp_url
        self.chunk_type = chunk_type
        self.chunk_args = chunk_args
        self.output_dir = output_dir
        self.cap = None
        self.frame_buffer = []
        self.buffer_start_time = None
        self.recording = False  # Flag to track when to save frames
        self.no_person_frame_count = 0  # Counter for consecutive frames without a person
        self.max_no_person_frames = 30  # Threshold before stopping the recording

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sliding_window_chunk(self, frame, current_time) -> Tuple[str, str]:
        """Sliding window chunking with overlap."""
        window_size = self.chunk_args.get("window_size", 85)
        fps = self.chunk_args.get("fps", 15)
        overlap = self.chunk_args.get("overlap", 0)

        if not self.frame_buffer:
            self.buffer_start_time = current_time

        self.frame_buffer.append(frame)

        if len(self.frame_buffer) >= window_size:
            formatted_time = datetime.utcfromtimestamp(self.buffer_start_time).strftime('%Y-%m-%d_%H-%M-%S')
            chunk_filename = f"chunk_{formatted_time}.avi"
            chunk_path = os.path.join(self.output_dir, chunk_filename)
            self._save_video_chunk(self.frame_buffer[:window_size], chunk_path, fps)
            
            frames_to_remove = window_size - overlap
            if frames_to_remove > 0:
                self.frame_buffer = self.frame_buffer[frames_to_remove:]
                if self.frame_buffer:
                    self.buffer_start_time += frames_to_remove / fps
            2
            return chunk_path, formatted_time
        
        return None, None
    
    def _yolo_person_detected(self, frame) -> bool:
        """Detect humans using YOLOv8 model."""
        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes.data:
                class_id = int(box[5].item())  # Extract class ID
                #print(f"Class detected: {class_id}")
                if class_id == 0:  # YOLO class 0 is 'person'
                    #print('PERSON STILL DETECTED')
                    self.no_person_frame_count = 0  # Reset counter when a person is detected
                    return True
        
        # If no person is detected, increment the counter
        self.no_person_frame_count += 1
        print(f"No person detected. Frame count: {self.no_person_frame_count}/{self.max_no_person_frames}")

        # If 30 consecutive frames have no person, return False
        if self.no_person_frame_count >= self.max_no_person_frames:
            print("[INFO] No person detected for 30 frames. Not currently recording.")
            return False

        return True  # Continue recording since we haven't hit the threshold yet

    def _interval_trigger_chunk(self, frame, current_time) -> Tuple[str, str]:
        """YOLO-based human detection chunking."""
        fps = self.chunk_args.get("fps", 15)

        person_detected = self._yolo_person_detected(frame)

        if person_detected:
            if not self.recording:
                print('[PERSON DETECTED] STARTING RECORDING')
                self.recording = True
                self.frame_buffer = []  # Reset buffer when starting a new chunk
                self.buffer_start_time = current_time
            
            self.frame_buffer.append(frame)
        
        elif self.recording and not person_detected:
            # Stop recording and save chunk
            formatted_time = datetime.utcfromtimestamp(self.buffer_start_time).strftime('%Y-%m-%d_%H-%M-%S')
            chunk_filename = f"human_chunk_{formatted_time}.avi"
            chunk_path = os.path.join(self.output_dir, chunk_filename)
            self._save_video_chunk(self.frame_buffer, chunk_path, fps)
            
            self.recording = False
            return chunk_path, formatted_time
        
        return None, None

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load RTSP stream chunks as LangChain Documents."""
        print(f"[INFO] Starting RTSP stream from: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            print("[ERROR] Failed to open RTSP stream.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] Stream ended or error reading frame.")
                break

            current_time = time.time()

            if self.chunk_type == "sliding_window":
                chunk_path, formatted_time = self._sliding_window_chunk(frame, current_time)
            elif self.chunk_type == "interval_trigger":
                chunk_path, formatted_time = self._interval_trigger_chunk(frame, current_time)
            else:
                raise ValueError("[ERROR] Invalid chunk type. Choose 'sliding_window' or 'interval_trigger'.")

            if chunk_path and formatted_time:
                yield Document(
                    page_content=f"Processed RTSP chunk saved at {chunk_path}",
                    metadata={
                        "chunk_path": chunk_path,
                        "timestamp": formatted_time,
                        "source": self.rtsp_url,
                    },
                )
                executor.submit(run_vlm_inference, chunk_path, formatted_time)

        self.cap.release()
        print("[INFO] RTSP stream processing complete.")

    def _save_video_chunk(self, frames: List, output_path: str, fps: int):
        """Save a list of frames as a video chunk."""
        if not frames:
            return
            
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"[INFO] Saved video chunk: {output_path}")

if __name__ == "__main__":
    output_dir = "rtsp_live_chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Test Sliding Window Chunking
    # print("\n[TEST] Running Sliding Window Chunking...")
    # loader_sw = RTSPChunkLoader(
    #     rtsp_url="rtsp://admin:setuq-h1B.24@192.168.1.28:554",
    #     chunk_type="sliding_window",
    #     chunk_args={
    #         "window_size": 85,
    #         "fps": 15,
    #         "overlap": 15,
    #     },
    #     output_dir=output_dir,
    # )

    # for doc in loader_sw.lazy_load():
    #     print(f"Sliding Window Chunk: {doc.metadata}")

    # Test Scene Change Interval Chunking
    print("\n[TEST] Running Yolo person detection Interval...")
    loader = RTSPChunkLoader(
        rtsp_url="rtsp://localhost:8554/live",
        chunk_type="interval_trigger",
        chunk_args={
            "fps": 30,
        },
        output_dir="rtsp_live_chunks",
    )

    for doc in loader.lazy_load():
        print(f"YOLO-based Chunk: {doc.metadata}")


