import matplotlib.animation as animation
import os

def get_unique_filename(base_name, extension, step=1, last_n=None, directory="images"):
    """Generate a unique filename by appending a number if the file already exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    base_name = f"{base_name}_step{step}"
    if last_n is not None:
        base_name += f"_last{last_n}"
    
    counter = 1
    file_name = os.path.join(directory, f"{base_name}.{extension}")
    while os.path.exists(file_name):
        file_name = os.path.join(directory, f"{base_name}_{counter}.{extension}")
        counter += 1
    return file_name

def create_animation(data, base_name="created_animation", speed=30, step=1, last_n=None, display_last_frames=60):
    """Create and save an animation from a numpy array."""
    fig, ax = plt.subplots()
    
    # Hide the x and y axis descriptions
    ax.set_xticks([])
    ax.set_yticks([])

    # If last_n is specified, slice the data to include only the last n elements
    if last_n is not None:
        data = data[-last_n:]

    # Initial setup for the plot
    img = ax.imshow(data[0])

    def update(frame):
        """Update the image for each frame."""
        img.set_data(data[frame])
        return img,

    # Create the animation
    frames = list(range(0, len(data), step)) + [len(data) - 1] * display_last_frames
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

    # Save the animation as a GIF
    file_name = get_unique_filename(base_name, "gif", step, last_n)
    ani.save(file_name, writer='pillow', fps=speed)
    plt.close(fig)
    print(f"Animation saved as {file_name}")
