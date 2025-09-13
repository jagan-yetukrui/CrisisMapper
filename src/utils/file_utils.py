"""
File utilities for CrisisMapper.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union
import hashlib
import mimetypes
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_info(file_path: Union[str, Path]) -> Dict:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'error': 'File does not exist'}
    
    stat = file_path.stat()
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    # Calculate file hash
    file_hash = calculate_file_hash(file_path)
    
    return {
        'path': str(file_path),
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'mime_type': mime_type,
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'hash': file_hash,
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir()
    }


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash as hex string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def find_files(directory: Union[str, Path], 
               extensions: Optional[List[str]] = None,
               recursive: bool = True) -> List[Path]:
    """
    Find files in directory with optional extension filter.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if extensions:
        # Convert extensions to lowercase and ensure they start with '.'
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in extensions]
    
    files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(file_path)
    
    return sorted(files)


def copy_file(src: Union[str, Path], 
              dst: Union[str, Path],
              create_dirs: bool = True) -> Path:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        Path to copied file
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if create_dirs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src, dst)
    logger.info(f"Copied {src} to {dst}")
    
    return dst


def move_file(src: Union[str, Path], 
              dst: Union[str, Path],
              create_dirs: bool = True) -> Path:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Whether to create destination directories
        
    Returns:
        Path to moved file
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if create_dirs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(str(src), str(dst))
    logger.info(f"Moved {src} to {dst}")
    
    return dst


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if deleted successfully, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    try:
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def cleanup_directory(directory: Union[str, Path], 
                     pattern: str = "*",
                     dry_run: bool = False) -> List[Path]:
    """
    Clean up directory by removing files matching pattern.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        dry_run: If True, only list files that would be deleted
        
    Returns:
        List of files that were (or would be) deleted
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    files_to_delete = list(directory.glob(pattern))
    deleted_files = []
    
    for file_path in files_to_delete:
        if file_path.is_file():
            if dry_run:
                logger.info(f"Would delete: {file_path}")
                deleted_files.append(file_path)
            else:
                if delete_file(file_path):
                    deleted_files.append(file_path)
    
    return deleted_files


def get_directory_size(directory: Union[str, Path]) -> Dict:
    """
    Get directory size information.
    
    Args:
        directory: Directory path
        
    Returns:
        Dictionary with size information
    """
    directory = Path(directory)
    
    if not directory.exists():
        return {'error': 'Directory does not exist'}
    
    total_size = 0
    file_count = 0
    dir_count = 0
    
    for item in directory.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1
        elif item.is_dir():
            dir_count += 1
    
    return {
        'path': str(directory),
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
        'file_count': file_count,
        'dir_count': dir_count
    }
