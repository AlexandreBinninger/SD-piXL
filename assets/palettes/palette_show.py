from utils.files_utils import load_hex, collect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def plot_palette(palette, name):
    fig, ax = plt.subplots(figsize=(len(palette), 2))
    for i, color in enumerate(palette):
        rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='none', facecolor=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
        ax.add_patch(rect)

    plt.xlim(0, len(palette))
    plt.ylim(0, 1)
    ax.axis('off')
    plt.title(f'Palette {name}')
    plt.show()

def display_palette(hex_path):
    palette = load_hex(hex_path)
    plot_palette(palette, hex_path)

def display_palettes(hex_dir):
    palettes = collect(hex_dir, '.hex')
    for palette in palettes:
        display_palette(''.join(palette))

def main():
    parser = argparse.ArgumentParser(description='Display palettes')
    parser.add_argument('--hex_path', type=str, default=None, help='Path to the hex file')
    parser.add_argument('--hex_dir', type=str, default=None, help='Path to the hex directory')
    args = parser.parse_args()
    if args.hex_path is not None:
        display_palette(args.hex_path)
    elif args.hex_dir is not None:
        display_palettes(args.hex_dir)
    else:
        print('Please provide either a hex path or a hex directory')

if __name__ == '__main__':
    main()