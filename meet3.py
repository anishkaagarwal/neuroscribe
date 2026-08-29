"""
Professional Meeting Recorder
Automated recording for Zoom and Google Meet with audio-video merging
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

import pyautogui
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from pyvirtualdisplay import Display
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meeting_recorder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MeetingRecorder:
    """
    Automated meeting recorder for Zoom and Google Meet
    """
    
    def __init__(self, output_dir="recordings", headless=False, auto_merge=True, keep_temp_files=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.driver = None
        self.recording = False
        self.headless = headless
        self.display = None
        self.auto_merge = auto_merge
        self.keep_temp_files = keep_temp_files
        
        self.video_thread = None
        self.audio_thread = None
        
        self.current_video_file = None
        self.current_audio_file = None
        
        self.meeting_info = {
            'platform': None,
            'meeting_id': None,
            'start_time': None,
            'end_time': None,
            'output_file': None
        }
        
    def setup_browser(self, profile_path=None):
        """Initialize Chrome browser with media permissions.

        profile_path: Chrome user-data dir for a logged-in session. Falls back to
        the CHROME_PROFILE_PATH env var, else a fresh temporary Chrome profile.
        """
        logger.info("Setting up browser...")

        if profile_path is None:
            profile_path = os.getenv("CHROME_PROFILE_PATH") or None
        
        if self.headless:
            self.display = Display(visible=0, size=(1920, 1080))
            self.display.start()
        
        options = webdriver.ChromeOptions()
        
        # Media permissions
        options.add_argument('--use-fake-ui-for-media-stream')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--start-maximized')
        
        # Use existing profile for logged-in sessions
        if profile_path:
            options.add_argument(f'user-data-dir={profile_path}')
        
        prefs = {
            "profile.default_content_setting_values.media_stream_mic": 1,
            "profile.default_content_setting_values.media_stream_camera": 1,
            "profile.default_content_setting_values.notifications": 2
        }
        options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1920, 1080)
        logger.info("Browser setup complete")
        
    def join_zoom(self, meeting_url, name="Meeting Recorder", password=None):
        """
        Join Zoom meeting via web browser
        
        Args:
            meeting_url: Full Zoom meeting URL
            name: Display name for the recorder
            password: Meeting password if required
        """
        logger.info(f"Joining Zoom meeting: {meeting_url}")
        self.meeting_info['platform'] = 'zoom'
        self.meeting_info['start_time'] = datetime.now().isoformat()
        
        try:
            self.driver.get(meeting_url)
            time.sleep(3)
            
            # Handle "Open Zoom Meetings" popup - click browser option
            try:
                cancel_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, "Cancel"))
                )
                cancel_btn.click()
            except TimeoutException:
                pass
            
            # Click "Join from Your Browser"
            try:
                join_browser = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Join from Your Browser"))
                )
                join_browser.click()
                logger.info("Clicked 'Join from Browser'")
            except TimeoutException:
                logger.warning("'Join from Browser' button not found, may already be on join page")
            
            time.sleep(2)
            
            # Enter password if required
            if password:
                try:
                    pwd_input = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.ID, "inputpasscode"))
                    )
                    pwd_input.send_keys(password)
                    
                    pwd_btn = self.driver.find_element(By.ID, "joinBtn")
                    pwd_btn.click()
                    logger.info("Entered meeting password")
                    time.sleep(2)
                except TimeoutException:
                    pass
            
            # Enter name
            try:
                name_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "inputname"))
                )
                name_input.clear()
                name_input.send_keys(name)
                logger.info(f"Entered name: {name}")
            except TimeoutException:
                logger.warning("Name input field not found")
            
            # Turn off camera and audio before joining
            try:
                # Turn off video
                video_btn = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Turn off video']")
                if video_btn:
                    video_btn.click()
                    
                # Mute audio
                audio_btn = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Mute audio']")
                if audio_btn:
                    audio_btn.click()
                    
                logger.info("Turned off camera and microphone")
            except NoSuchElementException:
                pass
            
            # Click Join button
            try:
                join_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "joinBtn"))
                )
                join_btn.click()
                logger.info("Clicked Join button")
            except TimeoutException:
                logger.error("Join button not found")
                return False
            
            time.sleep(5)
            logger.info("Successfully joined Zoom meeting")
            return True
            
        except Exception as e:
            logger.error(f"Error joining Zoom meeting: {e}")
            return False
    
    def join_google_meet(self, meeting_url, name="Meeting Recorder"):
        """
        Join Google Meet via web browser
        Requires being logged into Google account
        
        Args:
            meeting_url: Full Google Meet URL
            name: Display name (uses Google account name)
        """
        logger.info(f"Joining Google Meet: {meeting_url}")
        self.meeting_info['platform'] = 'google_meet'
        self.meeting_info['start_time'] = datetime.now().isoformat()
        
        try:
            self.driver.get(meeting_url)
            time.sleep(4)
            
            # Dismiss any popups
            try:
                dismiss_btn = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-label='Dismiss']"))
                )
                dismiss_btn.click()
            except TimeoutException:
                pass
            
            # Turn off camera
            try:
                camera_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-label*='camera']"))
                )
                # Check if camera is on (needs to be clicked)
                if "Turn off" in camera_btn.get_attribute("aria-label"):
                    camera_btn.click()
                    logger.info("Turned off camera")
                time.sleep(1)
            except TimeoutException:
                logger.warning("Camera button not found")
            
            # Turn off microphone
            try:
                mic_btn = self.driver.find_element(By.CSS_SELECTOR, "[aria-label*='microphone']")
                if "Turn off" in mic_btn.get_attribute("aria-label"):
                    mic_btn.click()
                    logger.info("Turned off microphone")
                time.sleep(1)
            except NoSuchElementException:
                logger.warning("Microphone button not found")
            
            # Click "Join now" or "Ask to join"
            try:
                join_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Join') or contains(., 'Ask to join')]"))
                )
                join_btn.click()
                logger.info("Clicked join button")
            except TimeoutException:
                logger.error("Join button not found")
                return False
            
            time.sleep(5)
            logger.info("Successfully joined Google Meet")
            return True
            
        except Exception as e:
            logger.error(f"Error joining Google Meet: {e}")
            return False
    
    def start_recording(self, duration=None):
        """
        Start recording video and audio
        
        Args:
            duration: Recording duration in seconds (None for indefinite)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        platform = self.meeting_info['platform']
        
        self.current_video_file = self.output_dir / f"{platform}_{timestamp}_video.avi"
        self.current_audio_file = self.output_dir / f"{platform}_{timestamp}_audio.wav"
        
        logger.info(f"Starting recording: {self.current_video_file} and {self.current_audio_file}")
        
        self.recording = True
        
        # Start video recording in separate thread
        self.video_thread = threading.Thread(
            target=self._record_screen,
            args=(str(self.current_video_file), duration)
        )
        self.video_thread.start()
        
        # Start audio recording in separate thread
        self.audio_thread = threading.Thread(
            target=self._record_audio,
            args=(str(self.current_audio_file), duration)
        )
        self.audio_thread.start()
        
        logger.info("Recording started successfully")
        
    def _record_screen(self, output_file, duration=None):
        """Internal method to record screen video"""
        logger.info(f"Screen recording started: {output_file}")
        
        screen_size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_file, fourcc, 20.0, screen_size)
        
        start_time = time.time()
        
        try:
            while self.recording:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Capture screenshot
                img = pyautogui.screenshot()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, screen_size)
                out.write(frame)
                
        except Exception as e:
            logger.error(f"Error during screen recording: {e}")
        finally:
            out.release()
            logger.info("Screen recording stopped")
    
    def _record_audio(self, output_file, duration=None, samplerate=44100):
        """Internal method to record system audio with improved buffering"""
        logger.info(f"Audio recording started: {output_file}")
        recording = []
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            if self.recording:
                recording.append(indata.copy())
    
        try:
            # Use larger blocksize to prevent buffer overflow
            # and enable latency='high' for more stable recording
            with sd.InputStream(
                samplerate=samplerate, 
                channels=2, 
                callback=callback,
                blocksize=4096,  # Increased from default
                latency='high'   # More stable, less prone to overflow
            ):
                start_time = time.time()
                while self.recording:
                    if duration and (time.time() - start_time) > duration:
                        break
                    time.sleep(0.1)
        
            if recording:
                data = np.concatenate(recording, axis=0)
                sf.write(output_file, data, samplerate)
                logger.info(f"Audio saved: {output_file}")
            else:
                logger.warning("No audio data recorded")
        
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def merge_audio_video(self, video_path, audio_path, output_path):
        """
        Merge audio and video files into a single MP4 file
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path for merged output file
        """
        logger.info(f"Merging audio and video...")
        logger.info(f"Video: {video_path}")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Output: {output_path}")
        
        # Verify input files exist
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        # Check file sizes
        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_path)
        logger.info(f"Video file size: {video_size / (1024*1024):.2f} MB")
        logger.info(f"Audio file size: {audio_size / (1024*1024):.2f} MB")
        
        if video_size == 0:
            logger.error("Video file is empty!")
            return False
        if audio_size == 0:
            logger.error("Audio file is empty!")
            return False
        
        try:
            logger.info("Loading video file...")
            video = VideoFileClip(str(video_path))
            logger.info(f"Video loaded: duration={video.duration}s, fps={video.fps}, size={video.size}")
            
            logger.info("Loading audio file...")
            audio = AudioFileClip(str(audio_path))
            logger.info(f"Audio loaded: duration={audio.duration}s")
            
            # Combine audio with video
            logger.info("Combining audio with video...")
            final = video.set_audio(audio)
            
            # Export result
            logger.info(f"Writing output file to: {output_path}")
            logger.info("This may take several minutes depending on video length...")
            
            final.write_videofile(
                str(output_path), 
                codec="libx264", 
                audio_codec="aac",
                verbose=True,  # Show progress
                threads=4  # Use multiple threads for faster encoding
            )
            
            # Verify output file was created
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                logger.info(f"Output file created: {output_size / (1024*1024):.2f} MB")
            else:
                logger.error("Output file was not created!")
                video.close()
                audio.close()
                final.close()
                return False
            
            # Close clips to release resources
            logger.info("Closing video clips...")
            video.close()
            audio.close()
            final.close()
            
            logger.info(f"✅ Video and audio successfully combined! Output: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging audio and video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def stop_recording(self):
        """Stop all recordings and merge files"""
        logger.info("Stopping recording...")
        self.recording = False
        
        # Wait for threads to finish
        logger.info("Waiting for recording threads to finish...")
        if self.video_thread:
            self.video_thread.join(timeout=15)
            if self.video_thread.is_alive():
                logger.warning("Video thread did not finish in time")
        if self.audio_thread:
            self.audio_thread.join(timeout=15)
            if self.audio_thread.is_alive():
                logger.warning("Audio thread did not finish in time")
        
        self.meeting_info['end_time'] = datetime.now().isoformat()
        logger.info("Recording stopped")
        
        # Give file system time to finalize writes
        time.sleep(2)
        
        # Merge audio and video if enabled
        if self.auto_merge and self.current_video_file and self.current_audio_file:
            # Verify files exist before attempting merge
            if not os.path.exists(self.current_video_file):
                logger.error(f"Video file not found: {self.current_video_file}")
                logger.error("Skipping merge")
            elif not os.path.exists(self.current_audio_file):
                logger.error(f"Audio file not found: {self.current_audio_file}")
                logger.error("Skipping merge")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                platform = self.meeting_info['platform']
                output_file = self.output_dir / f"{platform}_{timestamp}_final.mp4"
                
                logger.info(f"Starting merge process...")
                merge_success = self.merge_audio_video(
                    self.current_video_file,
                    self.current_audio_file,
                    output_file
                )
                
                if merge_success:
                    self.meeting_info['output_file'] = str(output_file)
                    logger.info(f"Merge successful! Final file: {output_file}")
                    
                    # Delete temporary files if requested
                    if not self.keep_temp_files:
                        try:
                            logger.info("Deleting temporary files...")
                            os.remove(self.current_video_file)
                            os.remove(self.current_audio_file)
                            logger.info("Temporary files deleted")
                        except Exception as e:
                            logger.warning(f"Could not delete temporary files: {e}")
                else:
                    logger.error("Merge failed! Keeping temporary files for debugging.")
                    self.meeting_info['video_file'] = str(self.current_video_file)
                    self.meeting_info['audio_file'] = str(self.current_audio_file)
        
        # Save meeting info
        self._save_meeting_info()
    
    def _save_meeting_info(self):
        """Save meeting metadata to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        info_file = self.output_dir / f"meeting_info_{timestamp}.json"
        
        with open(info_file, 'w') as f:
            json.dump(self.meeting_info, f, indent=2)
        
        logger.info(f"Meeting info saved: {info_file}")
    
    def leave_meeting(self):
        """Leave meeting and cleanup"""
        logger.info("Leaving meeting...")
        
        try:
            # Try to click leave button (platform-specific)
            if self.meeting_info['platform'] == 'zoom':
                leave_btn = self.driver.find_element(By.XPATH, "//button[contains(., 'Leave')]")
                leave_btn.click()
            elif self.meeting_info['platform'] == 'google_meet':
                leave_btn = self.driver.find_element(By.CSS_SELECTOR, "[aria-label*='Leave']")
                leave_btn.click()
        except NoSuchElementException:
            logger.warning("Leave button not found, closing browser directly")
        
        time.sleep(2)
        self.cleanup()
        
    def cleanup(self):
        """Close browser and cleanup resources"""
        logger.info("Cleaning up...")
        
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        if self.display:
            self.display.stop()
            self.display = None
        
        logger.info("Cleanup complete")


def main():
    """Example usage"""
    
    # Configuration
    MEETING_URL = "https://meet.google.com/pko-xamo-kbd"  # Replace with actual meeting URL
    MEETING_PLATFORM = "google_meet"  # or "zoom"
    MEETING_PASSWORD = None  # If required
    RECORDER_NAME = "Meeting Recorder Bot"
    RECORDING_DURATION = 3600  # 1 hour in seconds, or None for manual stop
    
    # Create recorder with auto-merge enabled
    recorder = MeetingRecorder(
        output_dir="recordings",
        auto_merge=True,  # Automatically merge audio and video
        keep_temp_files=False  # Delete temporary files after merging
    )
    
    try:
        # Setup browser
        recorder.setup_browser()
        
        # Join meeting
        if MEETING_PLATFORM == "zoom":
            success = recorder.join_zoom(MEETING_URL, RECORDER_NAME, MEETING_PASSWORD)
        elif MEETING_PLATFORM == "google_meet":
            success = recorder.join_google_meet(MEETING_URL, RECORDER_NAME)
        else:
            logger.error(f"Unknown platform: {MEETING_PLATFORM}")
            success = False
        
        if not success:
            logger.error("Failed to join meeting")
            return
        
        # Start recording
        recorder.start_recording(duration=RECORDING_DURATION)
        
        # Wait for recording to complete or manual interrupt
        if RECORDING_DURATION:
            time.sleep(RECORDING_DURATION)
        else:
            input("Press Enter to stop recording...")
        
        # Stop recording (will auto-merge if enabled)
        recorder.stop_recording()
        recorder.leave_meeting()
        
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
        recorder.stop_recording()
        recorder.cleanup()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        recorder.cleanup()


if __name__ == "__main__":
    main()