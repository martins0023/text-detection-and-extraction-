from cgitb import reset
from importlib import reload
from pdb import Restart
import tkinter
import tkinter.messagebox
from turtle import resetscreen
from click import command
import customtkinter
#sound/audio modules
from gtts import gTTS
import gtts
from playsound import playsound

#import the operating system
import os

#other modules
import argparse
import numpy
from textblob import TextBlob
import extract
import csv
from textblob import TextBlob
from langdetect import detect
import pycld2 as cld2

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def startup(self):
        os.system('python3 start.py')
        self.sidebar_button_2 = customtkinter.CTkLabel(self.sidebar_frame, text="succeeded..\ncheck directories")
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)

    def restart_p(self):
        os.system('python3 gui.py')
    

    def __init__(self):
        super().__init__()

        # configure window
        self.title("Text detection and extraction using python")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Text detection and extraction", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        #self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        #self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Get Started Now", command=self.startup)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        #self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        #self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="", state='disabled')
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self, text="EXIT", command=exit, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Text Image Extraction")
        self.tabview.add("Help")
        self.tabview.add("About Student")
        self.tabview.tab("Text Image Extraction").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Help").grid_columnconfigure(0, weight=1)
        self.tabview.tab("About Student").grid_columnconfigure(0, weight=1)

        #self.optionmenu_1 = customtkinter.CTkOptionMenu(self.tabview.tab("CTkTabview"), dynamic_resizing=False,
        #                                                values=["Value 1", "Value 2", "Value Long Long Long"])
        #self.optionmenu_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        #self.combobox_1 = customtkinter.CTkComboBox(self.tabview.tab("CTkTabview"),
        #                                            values=["Value 1", "Value 2", "Value Long....."])
        #self.combobox_1.grid(row=1, column=0, padx=20, pady=(10, 10))
        #self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Text Image Extraction"), text="Start detection",
        #                                                   command=self.open_input_dialog_event)
        #self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.label_tab_1 = customtkinter.CTkLabel(self.tabview.tab("Text Image Extraction"), text="Project Ouput \nin the folder directory;click \non the processed folder and open it,\n open the text_detected folder and \ncheck for the  text_detected.txt file.")
        self.label_tab_1.grid(row=0, column=0, padx=20, pady=20)


        self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("Help"), text="Project structure: \n\nThe program detects text \non an image given, processes it with \npre-installed and called modules \nwritten in the program to  \nextract the text on the image and\n saves it in a text file ")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)
        self.label_tab_2 = customtkinter.CTkLabel(self.tabview.tab("About Student"), text="NAME: Samuel Samuel\n MATRIC NO: HNDCOM000 \n DEPT: COMPUTER SCI \n SCHOOL: FCAH&PT \n SET: 2022/2023")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)

        # create display text for language detected
        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        
        

        self.label_tab_2 = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Note: please check \nthe folder directory\n for the output of\n the project. ")
        self.label_tab_2.grid(row=0, column=2, padx=10, pady=10)
        self.main_button_1 = customtkinter.CTkButton(master=self.radiobutton_frame, text="reload", command=lambda:exit(self.restart_p()), fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.main_button_1.grid(row=1, column=2, padx=10, pady=10)
        
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        #self.seg_button_1 = customtkinter.CTkSegmentedButton(self.slider_progressbar_frame)
        #self.seg_button_1.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_1.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_2 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_2.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        #self.slider_1 = customtkinter.CTkSlider(self.slider_progressbar_frame, from_=0, to=1, number_of_steps=4)
        #self.slider_1.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        self.slider_1 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="project description\n" + "An image-text detection and extraction using python3 .\nAIM: The aim of this project is to detect and extract texts \nfound on an image and save it a in text file \nHow to run;\n*Place the image to be detected in the home project folder directory, \nsave the image file as 'image.jpg' and click the 'get started now' \nbutton at the top left corner of the interface.")
        self.slider_1.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        
        #button

        #self.slider_2 = customtkinter.CTkButton(self.slider_progressbar_frame, orientation="vertical")
        #self.slider_2.grid(row=0, column=1, rowspan=5, padx=(10, 10), pady=(10, 10), sticky="ns")
        self.progressbar_3 = customtkinter.CTkProgressBar(self.slider_progressbar_frame, orientation="vertical")
        self.progressbar_3.grid(row=0, column=2, rowspan=5, padx=(10, 20), pady=(10, 10), sticky="ns")

        # set default values
        #read file text extracted output
        with open('processed/text_detected/text_detected.txt', "r") as f:
            contents = f.read()
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        #self.optionmenu_1.set("CTkOptionmenu")
        #self.combobox_1.set("CTkComboBox")
        self.slider_1.configure()
        #self.slider_2.configure(command=reload)
        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()
        self.textbox.insert("0.0", "FILE EXRACTED\n" + contents )
        #self.seg_button_1.configure(values=["CTkSegmentedButton", "Value 2", "Value 3"])
        #self.seg_button_1.set("Value 2")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Input your text:", title="Language Detection")
        dialogg = dialog.get_input()
        #print("CTkInputDialog:", dialogg)
        #save text in a file

        
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")


if __name__ == "__main__":
    app = App()
    app.mainloop()
