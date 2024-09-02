import cv2
from pathlib import Path
import tqdm

def create_video_from_array(img_arr_names:list,
                             video_name:str,
                             video_frame_freq: int = 1,
                             video_size = None,
                             fps: int = 30
                             ):
    img_arr = []
    print(f"Creating video from {len(img_arr_names)} images.")
    for i in tqdm.tqdm(range(len(img_arr_names))):
        filename = img_arr_names[i]
        if i % video_frame_freq == 0 or i == len(img_arr_names) - 1:
            img = cv2.imread(filename.as_posix())
            img_arr.append(img)
    
    if video_size is None:
        video_size = img_arr[0].shape[:2]
    else:
        for i in range(len(img_arr)):
            img_arr[i] = cv2.resize(img_arr[i], video_size, interpolation=cv2.INTER_NEAREST_EXACT)

    out = cv2.VideoWriter(
        video_name.as_posix(),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        video_size
    )
    for iii in range(len(img_arr)):
        out.write(img_arr[iii])
    out.release()

    print(f"video saved in '{video_name}'.")

import re
def numerical_sort(value):
    """
    Extracts numbers from a filename for sorting purposes.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_array_from_folder(folder_path:str, suffix="hard"):
    """
    Creates an array of image file names from a given folder based on a suffix.
    
    Args:
    - folder_path (str): Path to the folder containing the images.
    - suffix (str): Suffix to filter the images. Default is "hard".
    
    Returns:
    - List[Path]: List of paths to the image files.
    """
    folder = Path(folder_path)
    img_files = sorted(folder.glob(f"*_{suffix}.png"), key=lambda x: numerical_sort(x.stem))
    return img_files

def create_video_from_folder(folder_in : str, suffix: str, folder_out: str, video_frame_freq, video_size, fps):
    """
    Create the video from the given folder, with the given suffix
    """
    img_arr_names = create_array_from_folder(folder_in, suffix)
    create_video_from_array(img_arr_names, Path(folder_out) / f"video_{suffix}.mp4", video_frame_freq=video_frame_freq, video_size = video_size, fps=fps)
