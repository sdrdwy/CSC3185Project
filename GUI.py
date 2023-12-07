import PySimpleGUI as sg
import os.path
import Locating as L
import cv2
from Predictor import Predictor
import os
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse('浏览'),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]
image_viewer_column = [
    [
        sg.Image(key="-IMAGE1-"),
        sg.Image(key="-IMAGE3-"),
    ],
    [
        sg.Text("                 Origin Image                                                 "),
        sg.Text("Vessels"),
    ],
    [
        sg.Image(key="-IMAGE2-"),
        sg.Image(key="-IMAGE4-"),
    ],
    [
        sg.Text("                  Optic Disc                                                   "),
        sg.Text("Macula"),
    ],
    [
        sg.Text(size=(40, 1), key="-WHETHER-")
    ],
    [
        sg.Text(size=(40, 1), key="-WHAT-")
    ],
]
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]
window = sg.Window("Fudus Analysis Program", layout)
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  
        
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
        window["-IMAGE1-"].update(filename=filename)
        ## processed images
        cv2.imwrite("tmp/optic_disc.png", L.find_optic_disc(filename))
        window["-IMAGE2-"].update("tmp/optic_disc.png")
        cv2.imwrite("tmp/Vessels.png", L.find_Vessels(filename))
        window["-IMAGE3-"].update("tmp/Vessels.png")
        cv2.imwrite("tmp/find_macula.png", L.find_macula(filename))
        window["-IMAGE4-"].update("tmp/find_macula.png")
        pred = Predictor(model=os.getcwd()+"/model_cnn_best.pt")
        pred.load_label("Training_Set/RFMiD_Training_Labels.csv",["ID"])
        disease = pred.predict(filename)
        window["-WHAT-"].update("Possible Disease: " + disease, font=("黑体", 20), text_color="black")
            
window.close()