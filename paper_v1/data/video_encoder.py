"""
Encode videos at multiple bitrate levels for ABR simulation.
"""

import subprocess
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import shutil


class VideoEncoder:
    """Encode videos at different bitrate levels using FFmpeg."""
    
    # 6 bitrate levels (Kbps) - similar to YouTube/Netflix adaptive streaming
    BITRATE_LEVELS: List[int] = [300, 750, 1200, 1850, 2850, 6000]
    
    # Corresponding resolutions for each bitrate
    RESOLUTIONS: Dict[int, str] = {
        300: '426x240',    # 240p
        750: '640x360',    # 360p
        1200: '854x480',   # 480p
        1850: '1280x720',  # 720p
        2850: '1280x720',  # 720p high
        6000: '1920x1080'  # 1080p
    }
    
    # Supported input video formats
    SUPPORTED_INPUT_FORMATS = ['.mp4', '.y4m', '.mkv', '.avi', '.mov', '.webm', '.flv']
    
    def __init__(
        self, 
        input_dir: str = 'raw_videos',
        output_dir: str = 'encoded_videos'
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check FFmpeg availability
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found! Please install FFmpeg.")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed."""
        return shutil.which('ffmpeg') is not None
    
    def get_video_info(self, video_path: Path) -> Dict:
        """
        Extract video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = next(
                (s for s in data['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                return {}
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': float(data['format'].get('duration', 0)),
                'bitrate': int(data['format'].get('bit_rate', 0)) // 1000,  # Kbps
                'fps': eval(video_stream.get('r_frame_rate', '0/1'))
            }
        
        except Exception as e:
            print(f"⚠ Error getting video info: {e}")
            return {}
    
    def _has_audio_stream(self, video_path: Path) -> bool:
        """Check if video file has an audio stream."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except Exception:
            # If probe fails, assume no audio (safer for .y4m files)
            return False
    
    def encode_video(
        self,
        input_path: Path,
        output_path: Path,
        bitrate_kbps: int,
        resolution: str
    ) -> bool:
        """
        Encode a single video at specified bitrate and resolution.
        
        Args:
            input_path: Input video file
            output_path: Output video file
            bitrate_kbps: Target bitrate in Kbps
            resolution: Target resolution (e.g., '1280x720')
            
        Returns:
            True if encoding successful
        """
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',           # H.264 codec
            '-b:v', f'{bitrate_kbps}k',  # Video bitrate
            '-maxrate', f'{int(bitrate_kbps * 1.2)}k',  # Max bitrate
            '-bufsize', f'{int(bitrate_kbps * 2)}k',    # Buffer size
            '-vf', f'scale={resolution}',  # Resolution
            '-preset', 'medium',          # Encoding speed/quality tradeoff
            '-g', '48',                   # GOP size (2 seconds at 24fps)
            '-keyint_min', '48',          # Minimum GOP size
            '-sc_threshold', '0',         # Disable scene cut detection
            '-movflags', '+faststart',    # Web optimization
            '-y',                         # Overwrite output
            str(output_path)
        ]
        
        # Add audio encoding only if input has audio stream
        # Insert audio parameters before output path (insert in reverse order)
        if self._has_audio_stream(input_path):
            cmd.insert(-1, '-ar')
            cmd.insert(-1, '44100')      # Audio sample rate
            cmd.insert(-1, '-b:a')
            cmd.insert(-1, '128k')       # Audio bitrate
            cmd.insert(-1, '-c:a')
            cmd.insert(-1, 'aac')        # Audio codec
        else:
            cmd.insert(-1, '-an')        # No audio
        
        try:
            # Run FFmpeg with progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Simple progress indicator
            for line in process.stdout:
                if 'time=' in line:
                    print('.', end='', flush=True)
            
            process.wait()
            print()  # New line after dots
            
            if process.returncode == 0:
                return True
            else:
                print(f"✗ Encoding failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"✗ Encoding error: {e}")
            return False
    
    def _find_input_video(self, video_name: str) -> Path | None:
        """
        Find input video file with any supported extension.
        
        Args:
            video_name: Name of the video (without extension)
            
        Returns:
            Path to input video file, or None if not found
        """
        for ext in self.SUPPORTED_INPUT_FORMATS:
            input_path = self.input_dir / f"{video_name}{ext}"
            if input_path.exists():
                return input_path
        return None
    
    def encode_all_bitrates(self, video_name: str) -> Dict[int, Path]:
        """
        Encode a single video at all bitrate levels.
        
        Args:
            video_name: Name of the video (without extension)
            
        Returns:
            Dictionary mapping bitrate to output path
        """
        input_path = self._find_input_video(video_name)
        
        if not input_path:
            print(f"✗ Video not found: {video_name} (searched for: {', '.join(self.SUPPORTED_INPUT_FORMATS)})")
            return {}
        
        # Get original video info
        print(f"\n📹 Processing: {video_name}")
        info = self.get_video_info(input_path)
        
        if info:
            print(f"  Original: {info['width']}x{info['height']}, "
                  f"{info['bitrate']} Kbps, {info['duration']:.1f}s")
        
        # Create output directory for this video
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        encoded_files = {}
        
        # Encode at each bitrate level
        for idx, bitrate in enumerate(self.BITRATE_LEVELS, 1):
            resolution = self.RESOLUTIONS[bitrate]
            output_path = video_output_dir / f"{video_name}_{bitrate}kbps.mp4"
            
            print(f"\n[{idx}/{len(self.BITRATE_LEVELS)}] Encoding at {bitrate} Kbps ({resolution})")
            
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Already exists ({file_size_mb:.1f} MB) - skipping")
                encoded_files[bitrate] = output_path
                continue
            
            success = self.encode_video(
                input_path=input_path,
                output_path=output_path,
                bitrate_kbps=bitrate,
                resolution=resolution
            )
            
            if success:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Encoded successfully ({file_size_mb:.1f} MB)")
                encoded_files[bitrate] = output_path
            else:
                print(f"  ✗ Encoding failed")
        
        return encoded_files
    
    def encode_all_videos(self) -> Dict[str, Dict[int, Path]]:
        """
        Encode all videos in input directory.
        
        Returns:
            Nested dictionary: {video_name: {bitrate: path}}
        """
        # Check if input directory exists
        if not self.input_dir.exists():
            print(f"✗ Input directory does not exist: {self.input_dir.absolute()}")
            return {}
        
        if not self.input_dir.is_dir():
            print(f"✗ Input path is not a directory: {self.input_dir.absolute()}")
            return {}
        
        # List all files in directory for debugging
        all_files = list(self.input_dir.iterdir())
        if all_files:
            print(f"\n📂 Files found in input directory ({self.input_dir.absolute()}):")
            for f in sorted(all_files):
                file_type = "📁" if f.is_dir() else "📄"
                print(f"  {file_type} {f.name}")
        else:
            print(f"\n⚠ Input directory is empty: {self.input_dir.absolute()}")
        
        # Search for all supported video formats
        video_files = []
        for ext in self.SUPPORTED_INPUT_FORMATS:
            video_files.extend(list(self.input_dir.glob(f"*{ext}")))
        
        if not video_files:
            print(f"\n✗ No supported video files found in input directory: {self.input_dir.absolute()}")
            print(f"   Searched for: {', '.join(self.SUPPORTED_INPUT_FORMATS)}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"Video Encoding Pipeline")
        print(f"Input directory: {self.input_dir.absolute()}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Videos to encode: {len(video_files)}")
        print(f"Bitrate levels: {self.BITRATE_LEVELS}")
        print(f"{'='*60}")
        
        all_encoded = {}
        
        for video_file in video_files:
            video_name = video_file.stem
            encoded_files = self.encode_all_bitrates(video_name)
            
            if encoded_files:
                all_encoded[video_name] = encoded_files
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Encoding Summary:")
        print(f"  Videos processed: {len(all_encoded)}")
        total_files = sum(len(files) for files in all_encoded.values())
        print(f"  Total encoded files: {total_files}")
        print(f"{'='*60}\n")
        
        return all_encoded
    
    def get_encoding_summary(self) -> List[Dict]:
        """Get summary of all encoded videos."""
        summary = []
        
        for video_dir in self.output_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            video_name = video_dir.name
            bitrate_files = {}
            
            for encoded_file in video_dir.glob("*.mp4"):
                # Extract bitrate from filename
                try:
                    bitrate = int(encoded_file.stem.split('_')[-1].replace('kbps', ''))
                    size_mb = encoded_file.stat().st_size / (1024 * 1024)
                    bitrate_files[bitrate] = size_mb
                except:
                    continue
            
            if bitrate_files:
                summary.append({
                    'video': video_name,
                    'bitrates': bitrate_files,
                    'total_size_mb': sum(bitrate_files.values())
                })
        
        return summary


def main():
    """Main encoding script."""
    parser = argparse.ArgumentParser(
        description='Encode videos at multiple bitrate levels for ABR simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode all videos (default)
  python video_encoder.py
  
  # Encode a specific video
  python video_encoder.py --video bigbuckbunny
  
  # Encode a specific video with custom directories
  python video_encoder.py --video sintel --input-dir raw_videos --output-dir encoded_videos
        """
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Encode a specific video only (e.g., "bigbuckbunny")'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw_videos',
        help='Input directory containing raw videos (default: data/raw_videos)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/encoded_videos',
        help='Output directory for encoded videos (default: data/encoded_videos)'
    )
    
    args = parser.parse_args()
    
    print("\n🎬 Video Encoder for ABR Research\n")
    
    encoder = VideoEncoder(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Encode specific video or all videos
    if args.video:
        # Encode specific video
        print(f"⚙️  Encoding video: {args.video}")
        encoded_files = encoder.encode_all_bitrates(args.video)
        
        if encoded_files:
            print(f"\n✓ Encoding complete for: {args.video}")
            print(f"  Encoded: {len(encoded_files)} bitrate levels")
            
            total_size_mb = sum(
                path.stat().st_size / (1024 * 1024) 
                for path in encoded_files.values()
            )
            print(f"  Total size: {total_size_mb:.1f} MB")
            
            print("\nNext step: Calculate VMAF scores")
            print(f"  python vmaf_calculator.py --video {args.video}")
        else:
            print(f"\n✗ Failed to encode video: {args.video}")
    else:
        # Encode all videos
        encoded = encoder.encode_all_videos()
        
        if encoded:
            print("\n✓ Encoding complete!")
            print("\nSummary:")
            
            summary = encoder.get_encoding_summary()
            for item in summary:
                print(f"\n  📹 {item['video']}")
                print(f"     Encoded: {len(item['bitrates'])} bitrates")
                print(f"     Total size: {item['total_size_mb']:.1f} MB")
            
            print("\nNext step: Calculate VMAF scores")
            print("  python vmaf_calculator.py")
        else:
            print("\n✗ No videos were encoded")


if __name__ == '__main__':
    main()