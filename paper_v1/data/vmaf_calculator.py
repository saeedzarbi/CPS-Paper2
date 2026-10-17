"""
Calculate VMAF scores for encoded videos.
VMAF: Video Multimethod Assessment Fusion - perceptual video quality metric
"""

import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil


class VMAFCalculator:
    """Calculate VMAF scores comparing encoded videos to reference."""
    
    def __init__(
        self,
        reference_dir: str = 'raw_videos',
        encoded_dir: str = 'encoded_videos',
        output_dir: str = 'vmaf_scores'
    ):
        self.reference_dir = Path(reference_dir)
        self.encoded_dir = Path(encoded_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check FFmpeg with libvmaf
        if not self._check_vmaf_support():
            raise RuntimeError(
                "FFmpeg with libvmaf not found!\n"
                "Install: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)"
            )
    
    def _check_vmaf_support(self) -> bool:
        """Check if FFmpeg supports VMAF filter."""
        if not shutil.which('ffmpeg'):
            return False
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-filters'],
                capture_output=True,
                text=True,
                check=True
            )
            return 'libvmaf' in result.stdout
        except:
            return False
    def calculate_vmaf(
        self,
        reference_video: Path,
        distorted_video: Path,
        output_json: Path,
        timeout: int = 900,  # 15 minutes default (increased from 5)
        fast_mode: bool = False  # Use faster settings (lower resolution, subsampling)
    ) -> Optional[float]:
        """
        Calculate VMAF score between reference and distorted video.
        Scales both to 1920x1080 for consistent comparison.
        
        Args:
            reference_video: Original high-quality video
            distorted_video: Encoded video to evaluate
            output_json: Path to save detailed VMAF output
            timeout: Timeout in seconds (default: 900 = 15 minutes)
            fast_mode: Use faster settings (720p, subsampling) - less accurate but 3-5x faster
            
        Returns:
            Mean VMAF score (0-100) or None if failed
        """
        # Adaptive resolution: Use 720p for fast mode (3-5x faster)
        # For bitrate < 1200, 720p is usually sufficient
        if fast_mode:
            target_resolution = "1280:720"  # 720p - much faster
            subsample = "2"  # Process every 2nd frame (2x faster)
            n_threads = 8  # Use more threads
        else:
            target_resolution = "1920:1080"  # 1080p - standard
            subsample = "1"  # Process all frames
            n_threads = 8  # Increased from 4
        
        vmaf_filter = (
            f"[0:v]scale={target_resolution}:flags=bicubic,setpts=PTS-STARTPTS[dist];"
            f"[1:v]scale={target_resolution}:flags=bicubic,setpts=PTS-STARTPTS[ref];"
            f"[dist][ref]libvmaf="
            f"log_fmt=json:"
            f"log_path={output_json}:"
            f"n_threads={n_threads}:"
            f"n_subsample={subsample}"  # Skip frames for speed
        )
        
        cmd = [
            'ffmpeg',
            '-i', str(distorted_video),    # Distorted first
            '-i', str(reference_video),    # Reference second  
            '-filter_complex', vmaf_filter,
            '-f', 'null',
            '-',
            '-y'  # Overwrite output
        ]
        
        try:
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                # Only show error if verbose debugging needed
                # print(f"    FFmpeg stderr: {result.stderr[-500:]}")
                return None
            
            # Parse VMAF output
            if output_json.exists():
                try:
                    with open(output_json, 'r') as f:
                        data = json.load(f)
                    
                    # Extract mean VMAF score
                    if 'pooled_metrics' in data:
                        vmaf_score = data['pooled_metrics']['vmaf']['mean']
                        return round(vmaf_score, 2)
                    elif 'VMAF score' in data:  # Older format
                        return round(data['VMAF score'], 2)
                except json.JSONDecodeError:
                    print(f"    ✗ Invalid JSON output")
                    return None
            
            return None
            
        except subprocess.TimeoutExpired:
            timeout_min = timeout // 60
            print(f"    ✗ Timeout after {timeout_min} minutes")
            return None
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:100]}")
            return None
    def calculate_for_video(
        self,
        video_name: str,
        bitrate_levels: List[int],
        timeout: int = 900,  # 15 minutes per bitrate
        fast_mode: bool = False  # Use fast mode for low bitrates
    ) -> Dict[int, float]:
        """
        Calculate VMAF for all bitrate levels of a video.
        
        Args:
            video_name: Name of the video
            bitrate_levels: List of bitrate levels to evaluate
            
        Returns:
            Dictionary mapping bitrate to VMAF score
        """
        reference_video = self.reference_dir / f"{video_name}.mp4"
        
        if not reference_video.exists():
            print(f"✗ Reference video not found: {reference_video}")
            return {}
        
        encoded_dir = self.encoded_dir / video_name
        if not encoded_dir.exists():
            print(f"✗ Encoded directory not found: {encoded_dir}")
            return {}
        
        # Create output directory for this video
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        vmaf_scores = {}
        
        print(f"\n📊 Calculating VMAF for: {video_name}")
        
        for idx, bitrate in enumerate(bitrate_levels, 1):
            encoded_video = encoded_dir / f"{video_name}_{bitrate}kbps.mp4"
            output_json = video_output_dir / f"vmaf_{bitrate}kbps.json"
            
            if not encoded_video.exists():
                print(f"  [{idx}/{len(bitrate_levels)}] {bitrate} Kbps - ✗ File not found")
                continue
            
            # Check if already calculated
            if output_json.exists():
                try:
                    with open(output_json, 'r') as f:
                        data = json.load(f)
                        vmaf_score = data['pooled_metrics']['vmaf']['mean']
                        vmaf_scores[bitrate] = round(vmaf_score, 2)
                        print(f"  [{idx}/{len(bitrate_levels)}] {bitrate} Kbps - "
                              f"✓ Cached (VMAF: {vmaf_scores[bitrate]:.2f})")
                        continue
                except:
                    pass
            
            print(f"  [{idx}/{len(bitrate_levels)}] {bitrate} Kbps - Calculating...", end=' ')
            
            # Use fast mode for low bitrates (< 1200 kbps) - they don't need 1080p precision
            use_fast = fast_mode or (bitrate < 1200)
            
            vmaf_score = self.calculate_vmaf(
                reference_video=reference_video,
                distorted_video=encoded_video,
                output_json=output_json,
                timeout=timeout,
                fast_mode=use_fast
            )
            
            if vmaf_score is not None:
                vmaf_scores[bitrate] = vmaf_score
                print(f"✓ VMAF: {vmaf_score:.2f}")
            else:
                print("✗ Failed")
        
        return vmaf_scores
    
    def calculate_all_videos(
        self,
        bitrate_levels: List[int] = [300, 750, 1200, 1850, 2850, 6000],
        parallel: bool = False,
        timeout: int = 900,  # 15 minutes per bitrate
        fast_mode: bool = False,  # Use fast mode (720p, subsampling) for speed
        video_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate VMAF for all videos and bitrates.
        
        Args:
            bitrate_levels: List of bitrate levels
            parallel: Use parallel processing (faster but more CPU)
            
        Returns:
            DataFrame with VMAF scores
        """
        # Find all videos
        video_dirs = [d for d in self.encoded_dir.iterdir() if d.is_dir()]
        if video_names:
            allow = set(video_names)
            video_dirs = [d for d in video_dirs if d.name in allow]
        
        if not video_dirs:
            print("✗ No encoded videos found")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print(f"VMAF Calculation Pipeline")
        print(f"Videos: {len(video_dirs)}")
        print(f"Bitrate levels: {bitrate_levels}")
        print(f"Parallel processing: {parallel}")
        print(f"Fast mode: {fast_mode} (720p + subsampling for low bitrates)")
        print(f"{'='*60}")
        
        all_results = []
        
        if parallel:
            # Parallel processing (faster)
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        self.calculate_for_video,
                        video_dir.name,
                        bitrate_levels,
                        timeout,
                        fast_mode
                    ): video_dir.name
                    for video_dir in video_dirs
                }
                
                for future in as_completed(futures):
                    video_name = futures[future]
                    try:
                        vmaf_scores = future.result()
                        for bitrate, score in vmaf_scores.items():
                            all_results.append({
                                'video': video_name,
                                'bitrate_kbps': bitrate,
                                'vmaf': score
                            })
                    except Exception as e:
                        print(f"✗ Error processing {video_name}: {e}")
        else:
            # Sequential processing (easier to debug)
            for video_dir in video_dirs:
                video_name = video_dir.name
                vmaf_scores = self.calculate_for_video(video_name, bitrate_levels, timeout, fast_mode)
                
                for bitrate, score in vmaf_scores.items():
                    all_results.append({
                        'video': video_name,
                        'bitrate_kbps': bitrate,
                        'vmaf': score
                    })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            # Sort by video and bitrate
            df = df.sort_values(['video', 'bitrate_kbps']).reset_index(drop=True)
            
            # Save to CSV (merge with existing rows for videos not in this batch)
            csv_path = self.output_dir / 'vmaf_summary.csv'
            if csv_path.exists() and video_names:
                existing = pd.read_csv(csv_path)
                existing = existing[~existing['video'].isin(df['video'].unique())]
                df = pd.concat([existing, df], ignore_index=True)
                df = df.sort_values(['video', 'bitrate_kbps']).reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            print(f"\n✓ VMAF summary saved to: {csv_path}")
        
        return df
    
    def get_vmaf_summary(self) -> pd.DataFrame:
        """Load or generate VMAF summary."""
        csv_path = self.output_dir / 'vmaf_summary.csv'
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            return self.calculate_all_videos()
    
    def print_summary(self, df: pd.DataFrame):
        """Print nice summary of VMAF scores."""
        if df.empty:
            print("No VMAF data available")
            return
        
        print(f"\n{'='*60}")
        print("VMAF Summary")
        print(f"{'='*60}\n")
        
        for video in df['video'].unique():
            video_df = df[df['video'] == video]
            print(f"📹 {video}:")
            
            for _, row in video_df.iterrows():
                bitrate = row['bitrate_kbps']
                vmaf = row['vmaf']
                
                # Quality indicator
                if vmaf >= 80:
                    quality = "🟢 Excellent"
                elif vmaf >= 60:
                    quality = "🟡 Good"
                elif vmaf >= 40:
                    quality = "🟠 Fair"
                else:
                    quality = "🔴 Poor"
                
                print(f"  {bitrate:4d} Kbps → VMAF: {vmaf:5.2f} {quality}")
            
            print()


def main():
    """Main VMAF calculation script."""
    parser = argparse.ArgumentParser(
        description='Calculate VMAF scores for encoded videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential processing (default, easier to debug)
  python vmaf_calculator.py
  
  # Calculate VMAF for a specific video
  python vmaf_calculator.py --video bigbuckbunny
  
  # Fast mode (3-5x faster, 720p + subsampling)
  python vmaf_calculator.py --fast
  
  # Parallel processing (faster, uses more CPU)
  python vmaf_calculator.py --parallel
  
  # Fast + Parallel (fastest option)
  python vmaf_calculator.py --fast --parallel
  
  # Specific video with fast mode
  python vmaf_calculator.py --video sintel --fast
  
  # Custom timeout (in seconds)
  python vmaf_calculator.py --timeout 1800
        """
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing (faster but uses more CPU)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=900,
        help='Timeout per bitrate in seconds (default: 900 = 15 minutes)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast mode: 720p resolution + frame subsampling (3-5x faster, slightly less accurate)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Calculate VMAF for a specific video only (e.g., "bigbuckbunny")'
    )
    parser.add_argument(
        '--videos',
        type=str,
        default=None,
        help='Comma-separated video slugs (default: all encoded dirs, or v19 roster if empty encoded set)'
    )
    
    args = parser.parse_args()
    
    print("\n📊 VMAF Calculator for ABR Research\n")
    
    calculator = VMAFCalculator(
        reference_dir='data/raw_videos',
        encoded_dir='data/encoded_videos',
        output_dir='data/vmaf_scores'
    )
    
    video_list = None
    if args.videos:
        video_list = [v.strip() for v in args.videos.split(',') if v.strip()]
    
    # Calculate VMAF for specific video or all videos
    if args.video:
        # Calculate for specific video
        print(f"⚙️  Settings: Video={args.video}, Fast={args.fast}, Timeout={args.timeout}s ({args.timeout//60} min)")
        bitrate_levels = [300, 750, 1200, 1850, 2850, 6000]
        vmaf_scores = calculator.calculate_for_video(
            video_name=args.video,
            bitrate_levels=bitrate_levels,
            timeout=args.timeout,
            fast_mode=args.fast
        )
        
        if vmaf_scores:
            # Create DataFrame from results
            all_results = []
            for bitrate, score in vmaf_scores.items():
                all_results.append({
                    'video': args.video,
                    'bitrate_kbps': bitrate,
                    'vmaf': score
                })
            df = pd.DataFrame(all_results)
            df = df.sort_values(['video', 'bitrate_kbps']).reset_index(drop=True)
            
            # Update CSV file (load existing, update, save)
            csv_path = calculator.output_dir / 'vmaf_summary.csv'
            if csv_path.exists():
                existing_df = pd.read_csv(csv_path)
                # Remove old entries for this video
                existing_df = existing_df[existing_df['video'] != args.video]
                # Append new results
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.sort_values(['video', 'bitrate_kbps']).reset_index(drop=True)
            
            df.to_csv(csv_path, index=False)
            print(f"\n✓ VMAF summary saved to: {csv_path}")
            
            calculator.print_summary(df[df['video'] == args.video])
            
            print("\n✓ VMAF calculation complete!")
        else:
            print("\n✗ No VMAF scores calculated")
    else:
        # Calculate for all videos (or an explicit subset)
        print(f"⚙️  Settings: Parallel={args.parallel}, Fast={args.fast}, "
              f"Timeout={args.timeout}s ({args.timeout//60} min), "
              f"Videos={video_list or 'all encoded'}")
        df = calculator.calculate_all_videos(
            parallel=args.parallel,
            timeout=args.timeout,
            fast_mode=args.fast,
            video_names=video_list,
        )
        
        if not df.empty:
            calculator.print_summary(df)
            
            print("\n✓ VMAF calculation complete!")
            print("\nNext step: Extract SI/TI features")
            print("  python src/data_preparation/si_ti_extractor.py")
        else:
            print("\n✗ No VMAF scores calculated")


if __name__ == '__main__':
    main()