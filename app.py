from tkinter import Tk, Canvas, Button
from PIL import Image, ImageDraw
from mnist import predict_image

# Global variables for mouse position
last_x, last_y = None, None

def start_drawing(event):
    """Start drawing when the left mouse button is pressed."""
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    """Draw a line to the current mouse position."""
    global last_x, last_y
    # Draw on the canvas
    # smooth storkes 
    canvas.create_line(last_x, last_y, event.x, event.y, smooth=True, capstyle="round", fill="black", width=20)
    # Draw on the PIL Image
    draw_image.line([last_x, last_y, event.x, event.y], fill="black", width=20)
    last_x, last_y = event.x, event.y

def clear_canvas():
    """Clear the entire canvas and reset the image."""
    canvas.delete("all")
    draw_image.rectangle([0, 0, canvas_width, canvas_height], fill="white")

def save_and_predict_image():
    """Predict the canvas drawing to an image file."""
    file_name = "./images/canvas_drawing.png"
    # save image of size 64x64 and greyscale
    resize = image.resize((64, 64)).convert("L")
    resize.save(file_name)
    predicted_output = f'Predicted Text: {predict_image(file_name)}'
    canvas.create_text(400, 50, text=predicted_output, font=("Arial", 24), fill="blue")

    print(f"Image saved as {file_name}")

# Initialize the main window
root = Tk()
root.title("Handwritten Digit Recoginition")

# Canvas dimensions
canvas_width, canvas_height = 1024, 1024

# Create a Canvas widget
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Create a PIL Image to save the drawing
image = Image.new("RGB", (canvas_width, canvas_height), "white")
draw_image = ImageDraw.Draw(image)

# Bind mouse events to the canvas
canvas.bind("<Button-1>", start_drawing)  # Left mouse button pressed
canvas.bind("<B1-Motion>", draw)         # Mouse drag while left button pressed

# Add buttons
button_frame = Canvas(root, height=50)
button_frame.pack()

clear_button = Button(button_frame, text="Clear Canvas", command=clear_canvas)
clear_button.pack(side="left", padx=10, pady=5)

predict_btn = Button(button_frame, text="Predict", command=save_and_predict_image)
predict_btn.pack(side="right", padx=10, pady=5)


# Run the application
root.mainloop()
