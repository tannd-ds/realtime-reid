import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='path to input video')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='path to output video')
    parser.add_argument('--fps',
                        type=int,
                        default=12,
                        help='output video FPS')
    return parser.parse_args()


def convert_to_fps(input_path, output_path, fps):
    # Open the input video file
    video = cv2.VideoCapture(input_path)

    # Get the original frame rate of the video
    original_fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval to achieve the desired fps
    if original_fps < fps:
        print(f'Input video is less than {fps} FPS. Exiting...')
        exit(1)
    frame_interval = int(original_fps / fps)

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(
        output_path, fourcc, fps,
        (int(video.get(3)), int(video.get(4)))
    )

    # Read and write frames until the end of the video
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Write the frame to the output video
        if frame_count % frame_interval == 0:
            output.write(frame)

        frame_count += 1

    # Release the video capture and writer objects
    video.release()
    output.release()


if __name__ == '__main__':
    args = parse_args()
    input_file = args.input
    output_file = args.output

    convert_to_fps(input_file, output_file, args.fps)
