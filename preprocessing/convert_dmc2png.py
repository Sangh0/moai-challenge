import mritopng
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert dcm format to png format')
    parser.add_argument('--original_path', type=str, required=True,
                        help='directory of original folder to convert dcm format to png format')
    parser.add_argument('--convert_path', type=str, required=True,
                        help='directory of converting folder to convert dcm format to png format'')
    args = parser.parse_args()

    mritopng.convert_folder(args.original_path, args.convert_path)

if __name__ == '__main__':
    main()