import subprocess
import os
from datetime import datetime
import logging
from tqdm import tqdm
import sys


class VideoFrameExtractor:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def check_ffmpeg(self):
        """Check if ffmpeg is installed"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return True
        except FileNotFoundError:
            self.logger.error("FFmpeg not found. Please install FFmpeg first.")
            return False

    def parse_fps(self, fps_str):
        """Parse FPS string to float, handling various formats"""
        try:
            if "/" in fps_str:
                # Handle fraction format (e.g., '30/1')
                num, den = map(int, fps_str.split("/"))
                return num / den
            else:
                # Handle decimal format
                return float(fps_str)
        except Exception as e:
            self.logger.error(f"Error parsing FPS '{fps_str}': {str(e)}")
            return 30.0  # Default to 30 FPS if parsing fails

    def extract_frames(self, url, output_dir="frames", num_frames=100, format="jpg"):
        """Extract frames using FFmpeg streaming"""
        if not self.check_ffmpeg():
            return 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frames_saved = 0

        try:
            # Get video duration using ffprobe
            duration_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                url,
            ]

            self.logger.info("Getting video duration...")
            duration_output = subprocess.run(
                duration_cmd, capture_output=True, text=True
            )
            duration = float(duration_output.stdout.strip())

            # Get video dimensions and fps
            stream_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0",
                url,
            ]

            self.logger.info("Getting video information...")
            stream_output = subprocess.run(stream_cmd, capture_output=True, text=True)
            width, height, fps_str = stream_output.stdout.strip().split(",")
            fps = self.parse_fps(fps_str)

            self.logger.info(f"Video dimensions: {width}x{height}")
            self.logger.info(f"Video duration: {duration:.2f} seconds")
            self.logger.info(f"FPS: {fps}")

            # Calculate timestamps for frame extraction
            total_frames = int(duration * fps)
            frame_interval = max(1, total_frames // num_frames)
            timestamps = [i * frame_interval / fps for i in range(num_frames)]

            self.logger.info(f"Extracting {num_frames} frames...")

            with tqdm(total=num_frames, desc="Extracting frames") as pbar:
                for i, timestamp in enumerate(timestamps):
                    if timestamp >= duration:
                        break

                    output_filename = os.path.join(
                        output_dir,
                        f"frame_{frames_saved:03d}_{datetime.now():%Y%m%d_%H%M%S}.{format.lower()}",
                    )

                    # Extract single frame using ffmpeg
                    cmd = [
                        "ffmpeg",
                        "-ss",
                        str(timestamp),
                        "-i",
                        url,
                        "-vframes",
                        "1",
                        "-q:v",
                        "2",  # High quality
                        "-y",  # Overwrite output file
                        "-v",
                        "quiet",
                        output_filename,
                    ]

                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        frames_saved += 1
                        pbar.update(1)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(
                            f"Error extracting frame at {timestamp}s: {str(e)}"
                        )
                        continue

            self.logger.info(
                f"Successfully extracted {frames_saved} frames to {output_dir}"
            )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error: {str(e)}")
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")

        return frames_saved


def main():
    # Configuration
    PRESIGNED_URL = sys.argv[1]
    OUTPUT_FORMAT = "jpg"  # or "png"

    extractor = VideoFrameExtractor()

    try:
        extractor.extract_frames(
            PRESIGNED_URL,
            output_dir="extracted",
            num_frames=100,
            format=OUTPUT_FORMAT,
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
