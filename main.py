import argparse
from accelerate.utils import set_seed

from utils.create_video import create_video_from_folder
from pipelines.SD_piXL import SD_piXL

from libs.engine import merge_and_update_config
from libs.utils.argparse import accelerate_parser


def parse_tuple(argument):
    try:
        ret = list(map(int, argument.split(',')))
        return ret
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be comma-separated integers, e.g., '32,32'")


def main(args):

    sd_pixl = SD_piXL(args)
    sd_pixl.run()
    
    if args.make_video:
        create_video_from_folder(sd_pixl.png_logs_dir, suffix="hard", folder_out=sd_pixl.results_path, video_frame_freq=args.video_frame_freq, video_size=args.video_size, fps=args.fps)
        create_video_from_folder(sd_pixl.png_logs_dir, suffix="soft", folder_out=sd_pixl.results_path, video_frame_freq=args.video_frame_freq, video_size=args.video_size, fps=args.fps)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description="pixel diffusion rendering",
        parents=[accelerate_parser()]
    )
    
    # Configuration
    parser.add_argument("-c", "--config",
                        required=True, type=str,
                        default="config.yaml",
                        help="YAML/YML file for configuration.")
    
    parser.add_argument("-pt", "--prompt", default=None, type=str)
    parser.add_argument("-npt", "--negative_prompt", default=None, type=str)
    parser.add_argument("--download", action="store_true",
                        help="download models from huggingface automatically.")
    parser.add_argument("--force_download", action="store_true",
                        help="force the models to be downloaded from huggingface.")
    parser.add_argument("--palette", default=None, type=str,
                        help="palette for rendering.")
    parser.add_argument("-knc", "--kmeans_nb_colors", default=None, type=int,
                        help="If kmeans is used, number of colors for kmeans.")
    parser.add_argument("--size", default=None, type=parse_tuple, help="Specify the size as a tuple, e.g., --size=32,32")
    parser.add_argument("--input_image", default=None, type=str,
                        help="input image for rendering.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose information.")
    
    # Video parameters
    parser.add_argument("-mv", "--make_video", action="store_true",
                        help="make a video of the rendering process.")
    parser.add_argument("-frame_freq", "--video_frame_freq",
                        default=1, type=int,
                        help="video frame control.")
    parser.add_argument("--video_size", default="512,512", type=parse_tuple)
    parser.add_argument("--fps", default=30, type=int)
    args = parser.parse_args()
    
    # Removing the prompt arguments if they are not provided will allow the config to be loaded
    if args.prompt is None:
        del args.prompt
    if args.negative_prompt is None:
        del args.negative_prompt

    args = merge_and_update_config(args)
    if args.palette not in [None, "None"]:
        args.generator.palette = args.palette
    if args.size not in [None, "None"]:
        print(args.size[0], args.size[1])
        args.generator.image_H = int(args.size[0])
        args.generator.image_W = int(args.size[1])
    if args.input_image not in [None, "None"]:
        args.image = args.input_image
    if args.kmeans_nb_colors is not None:
        args.generator.kmeans_nb_colors = args.kmeans_nb_colors

    set_seed(args.seed)
    main(args)
