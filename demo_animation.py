import numpy as np
from animation_creation import create_animation


def generate_moving_square_frames(num_frames=30, size=32, square_size=8):
    """Generate frames showing a square moving horizontally."""
    frames = np.zeros((num_frames, size, size))
    for i in range(num_frames):
        start = (i * (size - square_size)) // (num_frames - 1)
        frames[i, 8:8 + square_size, start:start + square_size] = 1
    return frames


def main():
    data = generate_moving_square_frames()
    create_animation(data, base_name="demo_square", speed=10, step=1)


if __name__ == "__main__":
    main()

