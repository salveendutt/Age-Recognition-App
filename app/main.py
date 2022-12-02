import tkinter as tk
import video_capture as vc
import picture_capture as pc
def open_camera():
    print('Opening Camera...')
    vc.start_video()
    
def open_picture():
    print('Opening Picture...')
    pc.start_picture()
    
if __name__ == '__main__':
    ''' 
    Creating the main window 
    '''
    
    window = tk.Tk()
    window.title('Age Recognition app')
    window.geometry("600x400")
    window.option_add("*font", "Mini 20 bold italic")
    
    tk.Label().pack()
    tk.Label(text="Choose Source:").pack()
    button = tk.Button(
        text = "Open Camera",
        width = 15,
        height = 3,
        # bg = 
        # fg = 
        command = open_camera
        ).pack()
    button = tk.Button(
        text = "Open Picture",
        width = 15,
        height = 3,
        # bg = 
        # fg = 
        command = open_picture
        ).pack()
    window.mainloop()