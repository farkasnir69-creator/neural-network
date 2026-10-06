from tkinter import *

root = Tk()
root.title("Canvas Example")

canvas = Canvas(root, width=200, height=60)
canvas.pack()

y = 30
canvas.create_rectangle(500, 20, 20, 29, fill='red') 
root.mainloop()
