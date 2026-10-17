"""
Extract SI/TI (Spatial/Temporal Information) features from videos.
SI: Spatial complexity (detail level)
TI: Temporal complexity (motion level)
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import cv2
import pandas as pd


class SITIExtractor:
    """Extract Spatial Information (SI) and Temporal Information (TI) from videos."""
    
    def __init__(
        self,
        video_dir: str = 'raw_videos',
        output_dir: str = 'content_features'
    ):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_si_frame(self, frame: np.ndarray) -> float:
        """
        Calculate Spatial Information for a single frame.
        SI = standard deviation of Sobel filtered frame
        
        Args:
            frame: Video frame (grayscale)
            
        Returns:
            SI value
        """
        # Apply Sobel filter to detect edges
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # SI is the standard deviation of Sobel filtered image
        si = np.std(sobel)
        
        return si
    
    def calculate_ti_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Temporal Information between two consecutive frames.
        TI = standard deviation of pixel differences
        
        Args:
            frame1: Previous frame (grayscale)
            frame2: Current frame (grayscale)
            
        Returns:
            TI value
        """
        # Calculate frame difference
        diff = frame2.astype(np.float64) - frame1.astype(np.float64)
        
        # TI is the standard deviation of the difference
        ti = np.std(diff)
        
        return ti
    
    def extract_si_ti(
        self,
        video_path: Path,
        max_frames: int = 300
    ) -> Tuple[float, float, Dict]:
        """
        Extract SI and TI from a video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (for speed)
            
        Returns:
            Tuple of (mean_SI, mean_TI, stats_dict)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return 0.0, 0.0, {}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame skip for sampling
        if total_frames > max_frames:
            frame_skip = total_frames // max_frames
        else:
            frame_skip = 1
        
        si_values = []
        ti_values = []
        
        prev_frame = None
        frame_count = 0
        processed = 0
        
        print(f"    Processing {video_path.name}...")
        print(f"    Total frames: {total_frames}, Sampling every {frame_skip} frames")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for sampling
            if frame_count % frame_skip != 0:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate SI
            si = self.calculate_si_frame(gray)
            si_values.append(si)
            
            # Calculate TI (need previous frame)
            if prev_frame is not None:
                ti = self.calculate_ti_frames(prev_frame, gray)
                ti_values.append(ti)
            
            prev_frame = gray
            processed += 1
            
            # Progress indicator
            if processed % 50 == 0:
                print(f"    Processed {processed} frames...", end='\r')
        
        cap.release()
        
        print(f"    ✓ Processed {processed} frames                    ")
        
        # Calculate statistics
        mean_si = np.mean(si_values) if si_values else 0.0
        mean_ti = np.mean(ti_values) if ti_values else 0.0
        
        stats = {
            'mean_si': round(mean_si, 2),
            'mean_ti': round(mean_ti, 2),
            'max_si': round(np.max(si_values), 2) if si_values else 0.0,
            'max_ti': round(np.max(ti_values), 2) if ti_values else 0.0,
            'std_si': round(np.std(si_values), 2) if si_values else 0.0,
            'std_ti': round(np.std(ti_values), 2) if ti_values else 0.0,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'total_frames': total_frames,
            'processed_frames': processed
        }
        
        return mean_si, mean_ti, stats
    
    def extract_all_videos(self, video_names: List[str] | None = None) -> pd.DataFrame:
        """
        Extract SI/TI for videos in directory (optionally filtered by slug list).
        
        Returns:
            DataFrame with SI/TI values
        """
        if video_names:
            video_files = []
            for name in video_names:
                p = self.video_dir / f"{name}.mp4"
                if p.exists():
                    video_files.append(p)
                else:
                    print(f"  [WARN] missing reference: {p}")
        else:
            video_files = list(self.video_dir.glob("*.mp4"))
        
        if not video_files:
            print("✗ No videos found")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print(f"SI/TI Extraction Pipeline")
        print(f"Videos: {len(video_files)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        results = []
        
        for idx, video_path in enumerate(video_files, 1):
            video_name = video_path.stem
            
            print(f"\n[{idx}/{len(video_files)}] {video_name}")
            
            # Check if already processed
            json_path = self.output_dir / f"{video_name}_siti.json"
            
            if json_path.exists():
                print(f"    ✓ Already processed, loading from cache")
                with open(json_path, 'r') as f:
                    stats = json.load(f)
            else:
                # Extract SI/TI
                mean_si, mean_ti, stats = self.extract_si_ti(video_path)
                
                # Save to JSON
                with open(json_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"    ✓ Saved to {json_path.name}")
            
            # Add to results
            results.append({
                'video': video_name,
                'mean_si': stats['mean_si'],
                'mean_ti': stats['mean_ti'],
                'max_si': stats['max_si'],
                'max_ti': stats['max_ti'],
                'std_si': stats['std_si'],
                'std_ti': stats['std_ti'],
                'complexity': self._classify_complexity(
                    stats['mean_si'],
                    stats['mean_ti']
                )
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Save summary (merge when processing a subset)
            csv_path = self.output_dir / 'siti_summary.csv'
            if csv_path.exists() and video_names:
                existing = pd.read_csv(csv_path)
                existing = existing[~existing['video'].isin(df['video'].unique())]
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Summary saved to: {csv_path}")
        
        return df
    
    def _classify_complexity(self, si: float, ti: float) -> str:
        """
        Classify video complexity based on SI/TI values.
        
        Args:
            si: Spatial Information
            ti: Temporal Information
            
        Returns:
            Complexity class
        """
        # Rough thresholds (can be adjusted)
        if si > 50 and ti > 20:
            return "High (Complex + Fast Motion)"
        elif si > 50:
            return "High-SI (Complex Details)"
        elif ti > 20:
            return "High-TI (Fast Motion)"
        elif si > 30 or ti > 10:
            return "Medium"
        else:
            return "Low (Simple)"
    
    def print_summary(self, df: pd.DataFrame):
        """Print nice summary of SI/TI values."""
        if df.empty:
            print("No SI/TI data available")
            return
        
        print(f"\n{'='*60}")
        print("SI/TI Summary")
        print(f"{'='*60}\n")
        
        for _, row in df.iterrows():
            print(f"📹 {row['video']}:")
            print(f"   SI (Spatial):  {row['mean_si']:6.2f} (detail level)")
            print(f"   TI (Temporal): {row['mean_ti']:6.2f} (motion level)")
            print(f"   Complexity: {row['complexity']}")
            print()


def main():
    """Main SI/TI extraction script."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract SI/TI content features for ABR videos.")
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Comma-separated video slugs (default: all *.mp4 in --video-dir)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/raw_videos",
        help="Directory with reference MP4 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/content_features",
        help="Directory for per-video JSON and siti_summary.csv",
    )
    args = parser.parse_args()

    video_list = None
    if args.videos:
        video_list = [v.strip() for v in args.videos.split(",") if v.strip()]

    print("\n📊 SI/TI Extractor for ABR Research\n")
    
    extractor = SITIExtractor(
        video_dir=args.video_dir,
        output_dir=args.output_dir
    )
    
    df = extractor.extract_all_videos(video_names=video_list)
    
    if not df.empty:
        extractor.print_summary(df)
        
        print("\n✓ SI/TI extraction complete!")
        print("\nNext step: Prepare network traces")
        print("  python src/data_preparation/network_trace_processor.py")
    else:
        print("\n✗ No SI/TI data extracted")


if __name__ == '__main__':
    main()