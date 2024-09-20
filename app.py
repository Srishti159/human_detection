import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import os
import cv2
import pygame
from moviepy.editor import VideoFileClip
import numpy as np

# Initialize Pygame
pygame.init()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp6/weights/best.pt')  # Change the path to your model

def detect_humans(file_path):
    try:
        results = model(file_path)
        output_path = results.pandas().xyxy[0]
        num_humans = len(output_path[output_path['name'] == 'person'])
        return results, num_humans, None
    except Exception as e:
        return None, 0, str(e)

def show_image_with_detections(img_path, results):
    img = Image.open(img_path)
    results.render()
    img_with_detections = Image.fromarray(results.ims[0])
    img_with_detections.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img_with_detections)
    panel.config(image=img_tk)
    panel.image = img_tk

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs('temp', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('temp/output.avi', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        results.render()
        annotated_frame = results.ims[0]

        out.write(annotated_frame)

    cap.release()
    out.release()

def play_video_with_pygame(video_path):
    clip = VideoFileClip(video_path)
    screen = pygame.display.set_mode((clip.size[0], clip.size[1]))
    pygame.display.set_caption("Video Preview")

    for frame in clip.iter_frames(fps=clip.fps, dtype="uint8"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        pygame.time.wait(int(1000 / clip.fps))

    pygame.quit()

def close_image_display():
    panel.config(image='')
    panel.image = None

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        results, num_humans, error = detect_humans(file_path)
        if error:
            messagebox.showerror("Error", error)
        else:
            show_image_with_detections(file_path, results)
            messagebox.showinfo("Detection Results", f"Number of humans detected: {num_humans}")
            close_image_display()

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        processing_label.config(text="Processing video...")
        app.update_idletasks()
        process_video(file_path)
        processing_label.config(text="")
        play_video_with_pygame('temp/output.avi')
        try:
            os.remove('temp/output.avi')  # Remove the processed video after displaying
        except PermissionError:
            messagebox.showerror("Error", "Could not delete the video file. Please close any open video players and try again.")

app = tk.Tk()
app.title("Human Detection")
app.geometry("600x500")

upload_image_btn = tk.Button(app, text="Upload Image", command=upload_image)
upload_image_btn.pack(pady=20)

upload_video_btn = tk.Button(app, text="Upload Video", command=upload_video)
upload_video_btn.pack(pady=20)

processing_label = tk.Label(app, text="", font=("Helvetica", 14))
processing_label.pack(pady=20)

panel = tk.Label(app)
panel.pack(pady=20)

app.mainloop()
