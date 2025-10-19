#!/usr/bin/env python3
"""
Now i am creating LinkedIn Screenshot Folder Manager
==================================

Now i am managing LinkedIn screenshot folders to prevent storage bloat:
- Keeps only the latest 5 LinkedIn screenshot folders
- Archives older folders to data/archive/linkedin_screenshots/
- Runs automatically after each automation cycle

All code is original work developed for this computer vision project.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

class LinkedInFolderManager:
    """Now i am managing LinkedIn screenshot folder cleanup and archiving"""
    
    def __init__(self, keep_latest: int = 5):
        self.keep_latest = keep_latest
        self.logger = logging.getLogger(__name__)
        
        # Now i am defining paths
        self.raw_dir = Path("data/raw")
        self.linkedin_dir = self.raw_dir / "linkedin_screenshots"
        self.archive_dir = Path("data/archive/linkedin_screenshots")
        
        # Now i am ensuring archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def get_linkedin_folders(self) -> List[Tuple[Path, float]]:
        """Now i am getting all LinkedIn screenshot folders with their modification times"""
        if not self.linkedin_dir.exists():
            return []
        
        folders = []
        for item in self.linkedin_dir.iterdir():
            if item.is_dir() and item.name.startswith("linkedin_screenshots_"):
                try:
                    mtime = item.stat().st_mtime
                    folders.append((item, mtime))
                except OSError as e:
                    self.logger.warning(f"Could not get stats for {item}: {e}")
        
        # Now i am sorting by modification time (newest first)
        folders.sort(key=lambda x: x[1], reverse=True)
        return folders
    
    def archive_folder(self, folder_path: Path) -> bool:
        """Now i am archiving a folder to the archive directory"""
        try:
            # Now i am creating archive path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{folder_path.name}_archived_{timestamp}"
            archive_path = self.archive_dir / archive_name
            
            # Now i am moving folder to archive
            shutil.move(str(folder_path), str(archive_path))
            
            self.logger.info(f"Archived folder: {folder_path.name} → {archive_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to archive {folder_path}: {e}")
            return False
    
    def cleanup_folders(self) -> dict:
        """Now i am cleaning up LinkedIn screenshot folders, keeping only the latest N"""
        result = {
            'total_folders': 0,
            'kept_folders': 0,
            'archived_folders': 0,
            'failed_archives': 0,
            'kept_folder_names': [],
            'archived_folder_names': []
        }
        
        try:
            # Now i am getting all LinkedIn folders
            folders = self.get_linkedin_folders()
            result['total_folders'] = len(folders)
            
            if len(folders) <= self.keep_latest:
                self.logger.info(f"Only {len(folders)} folders found, no cleanup needed")
                result['kept_folders'] = len(folders)
                result['kept_folder_names'] = [f.name for f, _ in folders]
                return result
            
            # Now i am keeping the latest folders
            folders_to_keep = folders[:self.keep_latest]
            folders_to_archive = folders[self.keep_latest:]
            
            # Now i am logging what we're keeping
            for folder_path, _ in folders_to_keep:
                result['kept_folder_names'].append(folder_path.name)
                result['kept_folders'] += 1
            
            # Now i am archiving older folders
            for folder_path, _ in folders_to_archive:
                if self.archive_folder(folder_path):
                    result['archived_folder_names'].append(folder_path.name)
                    result['archived_folders'] += 1
                else:
                    result['failed_archives'] += 1
            
            self.logger.info(f"Cleanup completed: {result['kept_folders']} kept, {result['archived_folders']} archived")
            
        except Exception as e:
            self.logger.error(f"Error during folder cleanup: {e}")
            result['error'] = str(e)
        
        return result
    
    def get_storage_info(self) -> dict:
        """Now i am getting storage information about LinkedIn folders"""
        folders = self.get_linkedin_folders()
        
        total_size = 0
        folder_sizes = []
        
        for folder_path, mtime in folders:
            try:
                folder_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
                total_size += folder_size
                folder_sizes.append({
                    'name': folder_path.name,
                    'size_mb': folder_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                self.logger.warning(f"Could not calculate size for {folder_path}: {e}")
        
        return {
            'total_folders': len(folders),
            'total_size_mb': total_size / (1024 * 1024),
            'folders': folder_sizes
        }


def main():
    """Now i am main function for testing the folder manager"""
    print(" LinkedIn Screenshot Folder Manager")
    print("=" * 50)
    
    # Now i am setting up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Now i am initializing manager
    manager = LinkedInFolderManager(keep_latest=5)
    
    # Now i am getting current storage info
    print("\nCurrent Storage Information:")
    print("-" * 30)
    storage_info = manager.get_storage_info()
    print(f"Total folders: {storage_info['total_folders']}")
    print(f"Total size: {storage_info['total_size_mb']:.2f} MB")
    
    if storage_info['folders']:
        print("\nFolder Details:")
        for folder in storage_info['folders']:
            print(f"  • {folder['name']}: {folder['size_mb']:.2f} MB (modified: {folder['modified']})")
    
    # Now i am running cleanup
    print(f"\nRunning cleanup (keeping latest {manager.keep_latest} folders)...")
    print("-" * 50)
    
    result = manager.cleanup_folders()
    
    print(f"\nCleanup Results:")
    print(f"  • Total folders found: {result['total_folders']}")
    print(f"  • Folders kept: {result['kept_folders']}")
    print(f"  • Folders archived: {result['archived_folders']}")
    print(f"  • Failed archives: {result['failed_archives']}")
    
    if result['kept_folder_names']:
        print(f"\nKept folders:")
        for name in result['kept_folder_names']:
            print(f"  • {name}")
    
    if result['archived_folder_names']:
        print(f"\nArchived folders:")
        for name in result['archived_folder_names']:
            print(f"  • {name}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    main()
