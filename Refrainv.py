##Refrapy - Seismic Refraction Data Analysis
##Refrainv - Data inversion
##Author: Victor Guedes, MSc
##E-mail: vjs279@hotmail.com
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.colors import is_color_like
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import Tk, ttk, Toplevel, Frame, Button, Menu,Label, filedialog, messagebox, PhotoImage, simpledialog, Entry, Canvas, Scrollbar, StringVar
from os import path, makedirs, getcwd, name
from obspy import read
from obspy.signal.filter import lowpass, highpass
from scipy.signal import resample
from scipy.interpolate import interp1d,griddata
from scipy.spatial import ConvexHull
from numpy import array, where, polyfit, linspace, meshgrid, column_stack, c_, savetxt, shape,reshape,concatenate, hstack, linalg, mean, sqrt, zeros, arange, linspace, square, sort, unique
from numpy import all as np_all
from Pmw import initialise, Balloon
import pygimli as pg
from pygimli.physics import TravelTimeManager

from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import pandas as pd
import json 
#own modules
from dtreader import dtreader
from gpu_utils import gpu_config, is_cuda_available, get_cuda_info, accelerate_griddata


class Refrainv(Tk):
    
    def __init__(self):
        
        super().__init__()
        self.geometry("1600x900")
        self.title('Refrapy - Refrainv v2.0.0')
        self.configure(bg = "#F0F0F0")
        self.resizable(1,1)

        #check if on windows with nt kernel:
        if "nt" in name:
            self.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
        # if not, use unix formats
        else:
            self.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")

        frame_toolbar = Frame(self)
        frame_toolbar.grid(row=0,column=0,sticky="EW")
        
        # Add a menu bar
        menubar = Menu(self)
        self.config(menu=menubar)

        # Project menu
        project_menu = Menu(menubar, tearoff=0)
        project_menu.add_command(label="New Project", command=self.createProject)
        project_menu.add_command(label="Load Project", command=self.loadProject)
        project_menu.add_separator()
        project_menu.add_command(label="Exit", command=self.kill)
        menubar.add_cascade(label="Project", menu=project_menu)

        # Data menu
        data_menu = Menu(menubar, tearoff=0)
        data_menu.add_command(label="Load Pick File", command=self.loadPick)
        menubar.add_cascade(label="Data", menu=data_menu)

        # Inversion menu
        inversion_menu = Menu(menubar, tearoff=0)
        inversion_menu.add_command(label="Run Time-Terms Inversion", command=self.runTimeTerms)
        inversion_menu.add_command(label="Run Tomography", command=self.runTomography)
        inversion_menu.add_command(label="Batch Tomography", command=self.batchTomography)
        menubar.add_cascade(label="Inversion", menu=inversion_menu)

        # Visualization menu
        viz_menu = Menu(menubar, tearoff=0)
        viz_menu.add_command(label="Show Fit", command=self.showFit)
        viz_menu.add_command(label="Show Velocity Mesh", command=self.showPgResult)
        viz_menu.add_command(label="3D View", command=self.build3d)
        menubar.add_cascade(label="Visualization", menu=viz_menu)

        # Settings menu
        settings_menu = Menu(menubar, tearoff=0)
        settings_menu.add_command(label="GPU Settings", command=self.showGPUSettings)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self.help)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Keep a minimal toolbar for most-used actions (optional)
        #self.createToolbar()

        self.protocol("WM_DELETE_WINDOW", self.kill)
        self.initiateVariables()        

        photo = PhotoImage(file="%s/images/ico_refrapy.gif"%getcwd())
        labelPhoto = Label(frame_toolbar, image = photo, width = 151)
        labelPhoto.image = photo
        labelPhoto.grid(row=0, column =0, sticky="W")
        self.statusLabel = Label(frame_toolbar, text = "Create or load a project to start", font=("Arial", 11))
        self.statusLabel.grid(row = 0, column = 19, sticky = "W")

        initialise(self)

        self.ico_newProject = PhotoImage(file="%s/images/ico_newProject.gif"%getcwd())
        self.ico_loadProject = PhotoImage(file="%s/images/ico_loadProject.gif"%getcwd())
        self.ico_openPick = PhotoImage(file="%s/images/ico_loadPicks.gif"%getcwd())
        self.ico_invTimeterms = PhotoImage(file="%s/images/vm.gif"%getcwd())
        self.ico_invTomo = PhotoImage(file="%s/images/tomogram.gif"%getcwd())
        self.ico_layerMode = PhotoImage(file="%s/images/camadas.gif"%getcwd())
        self.ico_clearLayers = PhotoImage(file="%s/images/limpar.gif"%getcwd())
        self.ico_layer1 = PhotoImage(file="%s/images/layer1.gif"%getcwd())
        self.ico_layer2 = PhotoImage(file="%s/images/layer2.gif"%getcwd())
        self.ico_layer3 = PhotoImage(file="%s/images/layer3.gif"%getcwd())
        self.ico_reset = PhotoImage(file="%s/images/fechar.gif"%getcwd())
        self.ico_plotOptions = PhotoImage(file="%s/images/ico_plotOptions.gif"%getcwd())
        self.ico_save = PhotoImage(file="%s/images/salvar.gif"%getcwd())
        self.ico_fit = PhotoImage(file="%s/images/ico_fit.gif"%getcwd())
        self.ico_mergeResults = PhotoImage(file="%s/images/ico_mergeResults.gif"%getcwd())
        self.ico_velmesh = PhotoImage(file="%s/images/ico_velmesh.gif"%getcwd())
        self.ico_3d = PhotoImage(file="%s/images/ico_3d.gif"%getcwd())
        self.ico_help = PhotoImage(file="%s/images/ico_help.gif"%getcwd())

        bt = Button(frame_toolbar,image = self.ico_newProject,command = self.createProject,width=25)
        bt.grid(row = 0, column = 1, sticky="W")
        bl = Balloon(self)
        bl.bind(bt,"Create new project path")
        
        bt = Button(frame_toolbar,image = self.ico_loadProject,command = self.loadProject,width=25)
        bt.grid(row = 0, column = 2, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Load project path")
        
        bt = Button(frame_toolbar, image = self.ico_openPick,command = self.loadPick)
        bt.grid(row = 0, column = 3, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Load pick file")

        bt = Button(frame_toolbar, image = self.ico_layerMode,command = self.layersInterpretation)
        bt.grid(row = 0, column = 4, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Enable/disable layer assignmet mode (time-terms inversion)")

        bt = Button(frame_toolbar, image = self.ico_layer1,command = self.assignLayer1)
        bt.grid(row = 0, column = 5, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Assign layer 1 (direct wave)")

        bt = Button(frame_toolbar, image = self.ico_layer2,command = self.assignLayer2)
        bt.grid(row = 0, column = 6, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Assign layer 2 (refracted wave)")

        bt = Button(frame_toolbar, image = self.ico_layer3,command = self.assignLayer3)
        bt.grid(row = 0, column = 7, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Assign layer 3 (refracted wave)")

        bt = Button(frame_toolbar, image = self.ico_clearLayers,command = self.clearLayerAssignment)
        bt.grid(row = 0, column = 8, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Clear layer assignment")

        bt = Button(frame_toolbar, image = self.ico_invTimeterms,command = self.runTimeTerms)
        bt.grid(row = 0, column = 9, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Run time-terms inversion")

        bt = Button(frame_toolbar, image = self.ico_invTomo,command = self.runTomography)
        bt.grid(row = 0, column = 10, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Run tomography inversion")

        bt = Button(frame_toolbar, image = self.ico_fit,command = self.showFit)
        bt.grid(row = 0, column = 11, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Show model response (fit)")

        bt = Button(frame_toolbar, image = self.ico_velmesh,command = self.showPgResult)
        bt.grid(row = 0, column = 12, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Show tomography velocity model with mesh")

        bt = Button(frame_toolbar, image = self.ico_3d,command = self.build3d)
        bt.grid(row = 0, column = 13, sticky="W")
        b = Balloon(self)
        b.bind(bt,"3D view of velocity model")

        bt = Button(frame_toolbar, image = self.ico_save,command = self.saveResults)
        bt.grid(row = 0, column = 14, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Save results")

        bt = Button(frame_toolbar, image = self.ico_mergeResults,command = self.mergeResults)
        bt.grid(row = 0, column = 15, sticky="W")
        b = Balloon(self)
        b.bind(bt,"View layers (time-terms) on tomography model")

        bt = Button(frame_toolbar, image = self.ico_plotOptions,command = self.plotOptions)
        bt.grid(row = 0, column = 16, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Plot options")

        bt = Button(frame_toolbar, image = self.ico_reset,command = self.reset)
        bt.grid(row = 0, column = 17, sticky="W")
        b = Balloon(self)
        b.bind(bt,"Reset all")

        bt = Button(frame_toolbar,image=self.ico_help,command = self.help)
        bt.grid(row = 0, column = 18, sticky="W")
        bl = Balloon(self)
        bl.bind(bt,"Help")

        self.protocol("WM_DELETE_WINDOW", self.kill)
        self.initiateVariables()
        self.status_var = StringVar()
        self.status_var.set("Ready")
        status_bar = Label(self, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        status_bar.grid(row=99, column=0, sticky="EW")

    def batchTomography(self):
        """Run tomography inversion for a range of parameters with a dynamic batch count display."""
        # Dialog to get parameter ranges
        batchWindow = Toplevel(self)
        batchWindow.title('Refrainv - Batch Tomography')
        batchWindow.geometry("420x420")
        batchWindow.configure(bg="#F0F0F0")

        # Helper to parse and count steps
        def count_batches(*entries):
            try:
                lam_min, lam_max, lam_step = map(float, lam_entry.get().split(","))
                cell_min, cell_max, cell_step = map(float, cell_entry.get().split(","))
                zw_min, zw_max, zw_step = map(float, zweight_entry.get().split(","))
                lam_values = np.arange(lam_min, lam_max + lam_step, lam_step)
                cell_values = np.arange(cell_min, cell_max + cell_step, cell_step)
                zw_values = np.arange(zw_min, zw_max + zw_step, zw_step)
                total = len(lam_values) * len(cell_values) * len(zw_values)
                batch_count_label.config(text=f"Total batch runs: {total}")
            except Exception:
                batch_count_label.config(text="Total batch runs: -")

        # Title
        Label(batchWindow, text="Batch Tomography Inversion", font=("Arial", 13, "bold"), bg="#F0F0F0").pack(pady=(10, 5))

        # Smoothing (lam)
        frame_lam = Frame(batchWindow, bg="#F0F0F0")
        frame_lam.pack(pady=5, fill="x", padx=20)
        Label(frame_lam, text="Smoothing (lam) range (min,max,step):", bg="#F0F0F0").pack(side="left")
        lam_entry = Entry(frame_lam, width=15)
        lam_entry.pack(side="right")
        lam_entry.insert(0, "10,200,10")

        # Cell size
        frame_cell = Frame(batchWindow, bg="#F0F0F0")
        frame_cell.pack(pady=5, fill="x", padx=20)
        Label(frame_cell, text="Cell size range (min,max,step):", bg="#F0F0F0").pack(side="left")
        cell_entry = Entry(frame_cell, width=15)
        cell_entry.pack(side="right")
        cell_entry.insert(0, "2,10,2")

        # Zweight
        frame_zw = Frame(batchWindow, bg="#F0F0F0")
        frame_zw.pack(pady=5, fill="x", padx=20)
        Label(frame_zw, text="Vertical/horizontal smoothing (zweight) range (min,max,step):", bg="#F0F0F0").pack(side="left")
        zweight_entry = Entry(frame_zw, width=15)
        zweight_entry.pack(side="right")
        zweight_entry.insert(0, "0.1,0.5,0.1")

        # Dynamic batch count label
        batch_count_label = Label(batchWindow, text="Total batch runs: -", font=("Arial", 11, "bold"), fg="#0055aa", bg="#F0F0F0")
        batch_count_label.pack(pady=(10, 5))

        # Info
        Label(
            batchWindow,
            text="This will run a batched inversion with the program STANDARD parameters.\n"
             "If you have not provided these with a .json, internal standards will be used.",
            bg="#F0F0F0", wraplength=380, justify="center"
        ).pack(pady=(5, 10))

        # Bind events for dynamic update
        for entry in (lam_entry, cell_entry, zweight_entry):
            entry.bind("<KeyRelease>", lambda e: count_batches())
            entry.bind("<FocusOut>", lambda e: count_batches())

        # Initial count
        count_batches()

        # Run button
        Button(batchWindow, text="Run batch", font=("Arial", 11, "bold"), bg="#228B22", fg="white", command=lambda: runBatch()).pack(pady=15)

        def runBatch():
            try:
                lam_min, lam_max, lam_step = map(float, lam_entry.get().split(","))
                cell_min, cell_max, cell_step = map(float, cell_entry.get().split(","))
                zw_min, zw_max, zw_step = map(float, zweight_entry.get().split(","))
            except Exception as e:
                messagebox.showerror("Refrainv", f"Invalid input: {e}")
                return

            lam_values = np.arange(lam_min, lam_max + lam_step, lam_step)
            cell_values = np.arange(cell_min, cell_max + cell_step, cell_step)
            zw_values = np.arange(zw_min, zw_max + zw_step, zw_step)

            total_runs = len(lam_values) * len(cell_values) * len(zw_values)
            
            # Notify user about GPU usage in batch mode
            gpu_status = "GPU acceleration is ENABLED" if self.gpu_enabled else "GPU acceleration is DISABLED"
            if not messagebox.askyesno("Refrainv", f"Starting batch inversion with {total_runs} runs.\n\n{gpu_status}\n\nDo you want to continue?"):
                return
            
            progress = tqdm(total=total_runs, desc="Batch inversion")

            # Collect results for export
            results = []

            for lam in lam_values:
                for cell in cell_values:
                    for zw in zw_values:
                        # Set up mesh and inversion parameters
                        maxDepth = float(self.tomostandards["depth"])
                        paraDX = float(self.tomostandards["dx"])
                        paraMaxCellSize = cell
                        paraQuality = float(self.tomostandards["quality"])
                        self.tomoMesh = self.mgr.createMesh(
                            data=self.data_pg,
                            paraDepth=maxDepth,
                            paraDX=paraDX,
                            paraMaxCellSize=paraMaxCellSize,
                            quality=paraQuality
                        )
                        invert_kwargs = {
                            'data': self.data_pg,
                            'mesh': self.tomoMesh,
                            'verbose': False,
                            'lam': lam,
                            'zWeight': zw,
                            'useGradient': True,
                            'vTop': float(self.tomostandards["vtop"]),
                            'vBottom': float(self.tomostandards["vbottom"]),
                            'maxIter': int(self.tomostandards["maxiter"]),
                            'limits': [float(self.tomostandards["minvel"]), float(self.tomostandards["maxvel"])],
                            'secNodes': int(self.tomostandards["secnodes"])
                        }
                        # Optionally add start model logic here

                        vest = self.mgr.invert(**invert_kwargs)
                        chi2hist = self.mgr.inv.chi2History
                        chi2 = self.mgr.inv.chi2()
                        relrms = self.mgr.inv.relrms()
                        absrms = self.mgr.inv.absrms()
                        niter = self.mgr.inv.inv.iter()
                        # Save results with parameter info
                        param_str = f"lam{lam}_cell{cell}_zw{zw}"
                        outdir = os.path.join(self.projPath, "models", param_str)
                        self.mgr.saveResult(outdir)
                        os.makedirs(outdir, exist_ok=True)
                        np.savetxt(os.path.join(outdir, f"{self.lineName}_vel.txt"), vest)
                        # Save chi2 history as text
                        np.savetxt(os.path.join(outdir, f"{self.lineName}_chi2hist.txt"), chi2hist)
                        # Collect for summary table

                        results.append({
                            "lam": lam,
                            "cell": cell,
                            "zweight": zw,
                            "chi2": chi2,
                            "relrms": relrms,
                            "absrms": absrms,
                            "niter": niter,
                            "chi2hist": ";".join([str(x) for x in chi2hist])
                        })
                        progress.update(1)
            progress.close()
            # Export summary table as CSV
            df = pd.DataFrame(results)
            summary_path = os.path.join(self.projPath, "models", "batch_summary.csv")
            df.to_csv(summary_path, index=False)
            messagebox.showinfo("Refrainv", f"Batch inversion finished!\nSummary table saved to:\n{summary_path}")
            batchWindow.destroy()

    def help(self):

        helpWindow = Toplevel(self)
        helpWindow.title('Refrapick - Help')
        helpWindow.configure(bg = "#F0F0F0")
        helpWindow.resizable(0,0)
        #check if on windows with nt kernel:
        if "nt" in name:
            helpWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
        # if not, use unix formats
        else:
            helpWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
        #helpWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())

        Label(helpWindow, text = """Refrapy - Refrainv v2.0.0



        Refrainv provides tools for seismic refraction data inversion.

        If you use Refrapy in your work, please consider citing the following paper:

            Guedes, V.J.C.B., Maciel, S.T.R., Rocha, M.P., 2022. Refrapy: A Python program for seismic refraction data analysis,
            Computers and Geosciences. https://doi.org/10.1016/j.cageo.2021.105020.

    To report a bug and for more information, please visit github.com/viictorjs/Refrapy.


    Author: Victor Guedes, MSc
    E-mail: vjs279@hotmail.com
        """,font=("Arial", 11)).pack()
        helpWindow.tkraise()

    def showGPUSettings(self):
        """Show GPU configuration dialog."""
        gpuWindow = Toplevel(self)
        gpuWindow.title('Refrainv - GPU Settings')
        gpuWindow.configure(bg="#F0F0F0")
        gpuWindow.geometry("500x350")
        gpuWindow.resizable(False, False)
        
        # Set icon
        if "nt" in name:
            gpuWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
        else:
            gpuWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
        
        # Title
        Label(gpuWindow, text="GPU/CUDA Configuration", font=("Arial", 14, "bold"), bg="#F0F0F0").pack(pady=10)
        
        # Status frame
        status_frame = Frame(gpuWindow, bg="#F0F0F0", relief="sunken", bd=2)
        status_frame.pack(pady=10, padx=20, fill="x")
        
        # CUDA availability status
        cuda_info = get_cuda_info()
        Label(status_frame, text="CUDA Status:", font=("Arial", 11, "bold"), bg="#F0F0F0").pack(anchor="w", padx=10, pady=5)
        
        status_color = "green" if self.cuda_available else "red"
        status_text = "Available" if self.cuda_available else "Not Available"
        Label(status_frame, text=f"• {status_text}", font=("Arial", 10), bg="#F0F0F0", fg=status_color).pack(anchor="w", padx=20)
        
        # Device info
        if self.cuda_available:
            device_info = gpu_config.get_device_info()
            Label(status_frame, text="• Device Info:", font=("Arial", 10), bg="#F0F0F0").pack(anchor="w", padx=20)
            if isinstance(device_info, dict):
                Label(status_frame, text=f"  - Name: {device_info['name']}", font=("Arial", 9), bg="#F0F0F0").pack(anchor="w", padx=30)
                Label(status_frame, text=f"  - Compute Capability: {device_info['compute_capability']}", font=("Arial", 9), bg="#F0F0F0").pack(anchor="w", padx=30)
                Label(status_frame, text=f"  - Total Memory: {device_info['total_memory']}", font=("Arial", 9), bg="#F0F0F0").pack(anchor="w", padx=30)
        else:
            Label(status_frame, text=f"• {cuda_info}", font=("Arial", 9), bg="#F0F0F0", fg="red", wraplength=440).pack(anchor="w", padx=20, pady=5)
        
        # Current GPU setting
        current_frame = Frame(gpuWindow, bg="#F0F0F0")
        current_frame.pack(pady=10, padx=20, fill="x")
        
        Label(current_frame, text="Current Setting:", font=("Arial", 11, "bold"), bg="#F0F0F0").pack(anchor="w")
        current_status = "Enabled" if self.gpu_enabled else "Disabled"
        current_color = "green" if self.gpu_enabled else "gray"
        Label(current_frame, text=f"GPU Acceleration: {current_status}", font=("Arial", 10), bg="#F0F0F0", fg=current_color).pack(anchor="w", padx=20, pady=5)
        
        # Control buttons
        button_frame = Frame(gpuWindow, bg="#F0F0F0")
        button_frame.pack(pady=15)
        
        def enable_gpu():
            if not self.cuda_available:
                messagebox.showwarning("Refrainv", "CUDA is not available on this system.\n\nTo enable GPU support:\n1. Install CUDA toolkit\n2. Install CuPy: pip install cupy-cuda11x\n   (replace 11x with your CUDA version)")
                return
            
            if gpu_config.enable_gpu():
                self.gpu_enabled = True
                messagebox.showinfo("Refrainv", "GPU acceleration enabled!\n\nThe inversion process will now use CUDA when possible.")
                gpuWindow.destroy()
            else:
                messagebox.showerror("Refrainv", "Failed to enable GPU acceleration.")
        
        def disable_gpu():
            gpu_config.disable_gpu()
            self.gpu_enabled = False
            messagebox.showinfo("Refrainv", "GPU acceleration disabled.\n\nThe inversion process will use CPU only.")
            gpuWindow.destroy()
        
        # Enable/Disable buttons
        if self.cuda_available:
            if not self.gpu_enabled:
                Button(button_frame, text="Enable GPU", font=("Arial", 11, "bold"), 
                      bg="#228B22", fg="white", width=15, command=enable_gpu).pack(side="left", padx=5)
            else:
                Button(button_frame, text="Disable GPU", font=("Arial", 11, "bold"), 
                      bg="#DC143C", fg="white", width=15, command=disable_gpu).pack(side="left", padx=5)
        
        Button(button_frame, text="Close", font=("Arial", 11), 
              bg="#555555", fg="white", width=15, command=gpuWindow.destroy).pack(side="left", padx=5)
        
        # Information text
        info_frame = Frame(gpuWindow, bg="#F0F0F0")
        info_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        info_text = """Note: GPU acceleration can significantly speed up 
the inversion process for large datasets. However, 
it requires a CUDA-compatible GPU and properly 
configured CUDA drivers."""
        
        Label(info_frame, text=info_text, font=("Arial", 9), bg="#F0F0F0", 
              justify="left", wraplength=450).pack(anchor="w")
        
        gpuWindow.tkraise()
    
    def reset(self):

        if messagebox.askyesno("Refrainv", "Clear all?"):

            self.frame_plots.destroy()
            self.frame_data.destroy()
            self.frame_timeterms.destroy()
            self.frame_tomography.destroy()
            self.initiateVariables()
            self.statusLabel.configure(text="Create or load a project to start",font=("Arial", 11))
            messagebox.showinfo(title="Refrainv", message="All cleared successfully!")
        
    def initiateVariables(self):

        self.projReady = False
        self.xdata = []
        self.tdata = []
        self.sources = []
        self.dataArts = []
        self.data_sourcesArts = []
        self.data_pg = False
        self.tomoPlot = False
        self.timetermsPlot = False
        self.timetermsInv = False
        self.tomoMesh = False
        self.showRayPath = False
        self.rayPathColor = 'k'
        self.colormap = "jet_r"
        self.cmPlot = None
        self.coords_3d = []
        self.layerInterpretationMode = False
        self.layer2interpretate = 1
        self.layer1, self.layer2, self.layer3 = [],[],[]
        self.velocity1,self.velocity2,self.velocity3 = 0,0,0
        self.timeterms_response1_t, self.timeterms_response2_t, self.timeterms_response3_t = [],[],[]
        self.layer1_color = "r"
        self.layer2_color = "g"
        self.layer3_color = "b"
        self.showGrid = True
        self.data_color = "k"
        self.tomography_geohpones = False
        self.tomography_sources = False
        self.timeterms_geohpones = False
        self.timeterms_sources = False
        self.data_sources = True
        self.showGeophones = False
        self.geophonesPlot_timeterms = False
        self.geophonesPlot_tomography = False
        self.showSources = False
        self.sourcesPlot_timeterms = False
        self.sourcesPlot_tomography = False
        self.sourcesPlot_data = False
        self.dataLines = []
        self.new_x_tomography, self.new_y_tomography = False, False
        self.new_x_timeterms, self.new_y_timeterms = False, False
        self.tomography_3d_ready = False
        self.timeterms_3d_ready = False
        self.showMerged = False
        self.z2elev = False
        self.startModel = None
        self.startModelPath = None
        self.velModel = None
        self.__dict__.pop('tomostandards',None)
        
        # GPU configuration
        self.gpu_enabled = False
        self.cuda_available = is_cuda_available()
    
    def kill(self):

        out = messagebox.askyesno("Refrainv", "Do you want to close the software?")

        if out: plt.close('all');self.destroy(); self.quit()

    def assignLayer1(self):

        if self.layerInterpretationMode:
            
            self.layer2interpretate = 1    
            self.statusLabel.configure(text = 'Layer %d interpratation enabled!'%self.layer2interpretate)

    def assignLayer2(self):

        if self.layerInterpretationMode:
            
            self.layer2interpretate = 2 
            self.statusLabel.configure(text = 'Layer %d interpratation enabled!'%self.layer2interpretate)

    def assignLayer3(self):

        if self.layerInterpretationMode:
            
            self.layer2interpretate = 3   
            self.statusLabel.configure(text = 'Layer %d interpratation enabled!'%self.layer2interpretate)

    def mergeResults(self):

        if self.tomoPlot and self.timetermsPlot:

            if self.showMerged == False:
                
                if messagebox.askyesno("Refrainv", "Show layer(s) (from time-terms) over tomography model?"):

                    if self.layer2: self.merged_layer2, = self.ax_tomography.plot(self.gx_timeterms,self.z_layer2, c = "k")
                    if self.layer3: self.merged_layer3, = self.ax_tomography.plot(self.gx_timeterms,self.z_layer3, c = "k")
                    self.fig_tomography.canvas.draw()
                    self.showMerged = True
                    messagebox.showinfo(title="Refrainv", message="Layers displayed!")

            else:

                if messagebox.askyesno("Refrainv", "Remove layer(s) (from time-terms) over tomography model?"):

                    if self.layer2: self.merged_layer2.remove()
                    if self.layer3: self.merged_layer3.remove()
                    self.fig_tomography.canvas.draw()
                    self.showMerged = False
                    messagebox.showinfo(title="Refrainv", message="Layers removed!")

    def clearLayerAssignment(self):

        if messagebox.askyesno("Refrainv", "Clear layer interpretation?"):
                            
            del self.layer1[:]
            del self.layer2[:]
            del self.layer3[:]
            
            for i in range(len(self.sources)):
                
                for b in self.dataArts[i][self.sources[i]]:
                    
                    b.set_color("white")
                    b.set_edgecolor("k")

            self.fig_data.canvas.draw()
            messagebox.showinfo(title="Refrainv", message="Layer assignment was cleared!")
        
    def createPanels(self):
        
        self.frame_plots = Frame(self, bg = "white")
        self.frame_plots.grid(row = 1, column = 0, sticky = "NSWE")

        self.frame_data = Frame(self.frame_plots)
        self.frame_data.grid(row = 0, column = 0, sticky = "W", rowspan = 2)
        self.fig_data = plt.figure(figsize = (6,8.1))
        canvas_data = FigureCanvasTkAgg(self.fig_data, self.frame_data)
        canvas_data.draw()
        toolbar_data = NavigationToolbar2Tk(canvas_data, self.frame_data)
        toolbar_data.update()
        canvas_data._tkcanvas.pack()
        self.ax_data = self.fig_data.add_subplot(111)
        self.fig_data.patch.set_facecolor('#F0F0F0')
        self.ax_data.set_title("Observed data")
        self.ax_data.set_xlabel("POSITION [m]")
        self.ax_data.set_ylabel("TIME [s]")
        
        if self.showGrid: self.ax_data.grid(lw = .5, alpha = .5)
        
        self.ax_data.spines['right'].set_visible(False)
        self.ax_data.spines['top'].set_visible(False)
        self.ax_data.yaxis.set_ticks_position('left')
        self.ax_data.xaxis.set_ticks_position('bottom')
        
        self.frame_timeterms = Frame(self.frame_plots)
        self.frame_timeterms.grid(row = 0, column = 1, sticky = "NSWE")
        self.fig_timeterms = plt.figure(figsize = (9.5,3.7))
        canvas_timeterms = FigureCanvasTkAgg(self.fig_timeterms, self.frame_timeterms)
        canvas_timeterms.draw()
        toolbar_timeterms = NavigationToolbar2Tk(canvas_timeterms, self.frame_timeterms)
        toolbar_timeterms.update()
        canvas_timeterms._tkcanvas.pack()
        self.ax_timeterms = self.fig_timeterms.add_subplot(111)
        self.fig_timeterms.patch.set_facecolor('#F0F0F0')
        self.ax_timeterms.set_title("Time-terms velocity model")
        self.ax_timeterms.set_xlabel("POSITION [m]")
        self.ax_timeterms.set_ylabel("DEPTH [m]")
        
        if self.showGrid: self.ax_timeterms.grid(lw = .5, alpha = .5)
        else: self.ax_timeterms.grid(False)
        
        self.ax_timeterms.set_aspect("equal")
        self.ax_timeterms.spines['right'].set_visible(False)
        self.ax_timeterms.spines['top'].set_visible(False)
        self.ax_timeterms.yaxis.set_ticks_position('left')
        self.ax_timeterms.xaxis.set_ticks_position('bottom')

        self.frame_tomography = Frame(self.frame_plots)
        self.frame_tomography.grid(row = 1, column = 1, sticky = "NSWE")
        self.fig_tomography = plt.figure(figsize = (9.5,3.7))
        canvas_tomography = FigureCanvasTkAgg(self.fig_tomography, self.frame_tomography)
        canvas_tomography.draw()
        toolbar_tomography = NavigationToolbar2Tk(canvas_tomography, self.frame_tomography)
        toolbar_tomography.update()
        canvas_tomography._tkcanvas.pack()
        self.ax_tomography = self.fig_tomography.add_subplot(111)
        self.fig_tomography.patch.set_facecolor('#F0F0F0')
        self.ax_tomography.set_title("Tomography velocity model")
        self.ax_tomography.set_xlabel("POSITION [m]")
        self.ax_tomography.set_ylabel("DEPTH [m]")
        
        if self.showGrid: self.ax_tomography.grid(lw = .5, alpha = .5)
        else: self.ax_tomography.grid(False)

        self.ax_tomography.set_aspect("equal")
        self.ax_tomography.spines['right'].set_visible(False)
        self.ax_tomography.spines['top'].set_visible(False)
        self.ax_tomography.yaxis.set_ticks_position('left')
        self.ax_tomography.xaxis.set_ticks_position('bottom')

        self.frame_plots.tkraise()
        plt.close(self.fig_data)
        plt.close(self.fig_tomography)
        plt.close(self.fig_timeterms)
      
      
    def createProject(self):

        self.projPath = filedialog.askdirectory()
        
        if self.projPath:
            
            projName = simpledialog.askstring("Refrainv","Enter the name of the project to be created:")
            
            if not path.exists(self.projPath+"/"+projName):
                
                makedirs(self.projPath+"/"+projName)
                local = self.projPath+"/"+projName+"/"
                makedirs(local+"data")
                self.p_data = local+"data/"
                makedirs(local+"picks")
                self.p_picks = local+"picks/"
                makedirs(local+"models")
                self.p_models = local+"models/"
                makedirs(local+"gps")
                self.p_gps = local+"gps/"
                self.projPath = local
                self.projReady = True
                self.createPanels()
                messagebox.showinfo(title="Refrainv", message="Successfully created the project!")
                self.statusLabel.configure(text="Project path ready!",font=("Arial", 11))
                self.status_var.set("Project created successfully.")
            else:
                
                messagebox.showinfo(title="Refrainv", message="A project was detected, please choose another name or directory!")
        
    def loadProject(self):

        self.projPath = filedialog.askdirectory()
        
        if self.projPath:
            
            if path.exists(self.projPath+"/"+"data") and \
            path.exists(self.projPath+"/"+"picks") and \
            path.exists(self.projPath+"/"+"models") and \
            path.exists(self.projPath+"/"+"gps"):

                self.p_data = self.projPath+"/"+"data/"
                self.p_picks = self.projPath+"/"+"picks/"
                self.p_models = self.projPath+"/"+"models/"
                self.p_gps = self.projPath+"/"+"gps/"
                self.projReady = True
                self.createPanels()
                messagebox.showinfo(title="Refrainv", message="Successfully loaded the project path!")
                self.statusLabel.configure(text="Project path ready!",font=("Arial", 11))
                self.status_var.set("Project loaded successfully.")
            else: messagebox.showerror(title="Refrainv", message="Not all folders were detected!\nPlease, check the structure of the selected project.")

    def loadPick(self):
        
        if self.projReady:

            if self.data_pg:

                if messagebox.askyesno("Refrainv", "Load new data? (all current analysis have to be cleared)"): self.reset();

            if self.data_pg == False:    
                
                pickFile = filedialog.askopenfilename(title='Open', initialdir = self.projPath+"/picks/", filetypes=[('Pick file', '*.sgt'),('Protomo Tools','*.#dt')])
                self.lineName = path.basename(pickFile)[:-4]
            
                if path.basename(pickFile)[-4:]=='.#dt': 
                    dtreader( pickFile,self.lineName)
                    pickFile=pickFile[:-4]+'.sgt'
                if pickFile:

                    self.data_pg = pg.DataContainer(pickFile, 's g')

                    with open(pickFile, "r") as file:

                        lines = file.readlines()
                        npoints = int(lines[0].split()[0])
                        sgx = [float(i.split()[0]) for i in lines[2:2+npoints]]
                        sgz = [float(i.split()[1]) for i in lines[2:2+npoints]]
                        sgtindx = lines.index("#s g t\n")
                        s = [int(i.split()[0]) for i in lines[sgtindx+1:]]
                        g = [int(i.split()[1]) for i in lines[sgtindx+1:]]
                        t = [float(i.split()[2]) for i in lines[sgtindx+1:]]
                        sx = [sgx[i-1] for i in s]
                        gx = [sgx[i-1] for i in g]
                        gz = [sgz[i-1] for i in g]                    
                        self.gx = gx
                        self.gz = gz
                        self.sgx = sgx
                        self.sgz = sgz
                        #self.dx = self.gx[1]-self.gx[0]
                        self.dx=np.median(np.abs(np.diff(self.gx)))
                        self.sx, self.sz = [], []
                        
                        for i in list(set(s)):

                            self.sx.append(sgx[i-1])
                            self.sz.append(sgz[i-1])
                        print("what")
                        if any(z > 0 for z in gz): 
                            self.z2elev = True

                        if self.z2elev:

                            self.ax_timeterms.set_ylabel("ELEVATION [m]")
                            self.ax_tomography.set_ylabel("ELEVATION [m]")
                            self.fig_timeterms.canvas.draw()
                            self.fig_tomography.canvas.draw()
                        #saving computing time by setting sme apras outside of loop:
                        #seize=self.dx*4
                        seize=self.dx
                        #taking out some appends from below as append is slow
                        
                        self.sources=list(set(sx))
                        for i,src in enumerate(tqdm(list(set(sx)))):
        
                         #   self.sources.append(src)
                            self.xdata.append({src:[]})
                            self.tdata.append({src:[]})
                            self.dataArts.append({src:[]})
                            
                            if self.data_sources:

                                if self.showSources:
                                    #plots the source points
                                    sourcePlot = self.ax_data.scatter(src,0,c="y",edgecolor="k",s=100,marker="*",zorder=99)
                                    self.data_sourcesArts.append(sourcePlot)

                            for j,x in enumerate(gx):

                                if sx[j] == src:

                                    self.xdata[i][src].append(x)
                                    self.tdata[i][src].append(t[j])
                                    #dataPlot = self.ax_data.scatter(x,t[j],facecolors='w',s=seize,edgecolor=self.data_color,picker=self.dx,zorder=99)
                                    #self.dataArts[i][src].append(dataPlot)
                             #replace that awful loop: 
                            sxarray=np.array(sx)
                            gxarray=np.array(gx)
                            tarray=np.array(t)
                            all_src_occurence_indexes=np.where(sxarray==src)[0]
                            
                            all_x_occurences=gxarray[ all_src_occurence_indexes]
                            all_t_occurences=tarray[ all_src_occurence_indexes]
                            dataPlot=self.ax_data.scatter(all_x_occurences,all_t_occurences,facecolors='w',s=seize,edgecolor=self.data_color,picker=self.dx,zorder=99)
                            self.dataArts[i][src].append(dataPlot)                                
                                
                            dataLine, = self.ax_data.plot(self.xdata[i][src], self.tdata[i][src], c = self.data_color)
                            self.dataLines.append(dataLine)

                        self.fig_data.canvas.draw()
                        messagebox.showinfo(title="Refrainv", message="Traveltimes data have been loaded successfully!")
                        self.status_var.set("Project created successfully.")
    
    
    def runTimeTerms(self):

        if self.layer1 and self.layer2 or self.layer1 and self.layer3:
                    
            self.clearTimeTermsPlot()

            regw = simpledialog.askfloat("Refrainv", "Enter the regularization weight to be used for data inversion or cancel for default (lambda = 0.1)")
            
            if regw == None: regw = 0.1

            gx = list(set(self.gx))
            self.gx_timeterms = gx
            gz = self.gz[:len(gx)]
            self.gz_timeterms = gz
            
            def solve(layer, G, d, w):
                
                rMs = zeros((int(len(layer)), int(len(self.sources)+len(gx)+1)))
                r = 0
                
                for i in range(len(gx)):
                    
                    for j in range(len(self.sources)):
                        
                        if  self.sources[j] > min(gx) and self.sources[j] < max(gx): #se for fonte intermediaria
                            
                            if gx[i]+self.dx >= self.sources[j] and gx[i]-self.dx <= self.sources[j]:
                                
                                rMs[r][j] = w
                                rMs[r][len(self.sources)+i] = -w
                                r += 1
                                
                        elif self.sources[j] <= min(gx):
                            
                            if gx[i]-self.dx <= self.sources[j]:
                                
                                rMs[r][j] = w
                                rMs[r][len(self.sources)+i] = -w
                                r += 1
                                
                        elif self.sources[j] >= max(gx):
                            
                            if gx[i]+self.dx >= self.sources[j]:
                                
                                rMs[r][j] = w
                                rMs[r][len(self.sources)+i] = -w
                                r += 1
                                
                rMs = rMs[~np_all(rMs == 0, axis=1)] #regularization matrix of time-terms from sources
                rMg = zeros((int(len(layer)), int(len(self.sources)+len(gx)+1)))
                
                for i,j in zip(range(shape(rMg)[0]), range(shape(rMg)[1])):
                    
                    try:
                        
                        rMg[i][len(self.sources)+j] = w
                        rMg[i][len(self.sources)+j+1] = -w
                        
                    except: pass
                   
                rMg = rMg[~np_all(rMg == 0, axis=1)][:-2] #regularization matrix of time-terms from geophones
                rM = concatenate((rMs, rMg))
                rd = hstack((d, [i*0 for i in range(shape(rM)[0])]))
                rG = concatenate((G, rM))
                sol, sse, rank, sv = linalg.lstsq(rG, rd)
                
                return sol

            if self.layer1:
                
                d1 = array([self.layer1[i][1] for i in range(len(self.layer1))])
                G1 = array([self.layer1[i][3] for i in range(len(self.layer1))])
                G1 = reshape(G1, (len(G1),1))
                slowness  = []
                
                for time,delta in zip(d1,G1):
                    
                    if delta == 0: pass
                    else: slowness.append(time/delta)

                mean_slowness = mean(slowness)
                v1 = 1/mean_slowness
                self.velocity1 = v1
                list_ot1, list_pt1 = [],[]
                self.timeterms_response1_x = []
                self.timeterms_response1_t = []
                    
                for p in self.layer1:
                    
                    x = p[-3]
                    geop = p[0]
                    ot = p[1] #observed traveltime
                    pt = x/v1 #predicted traveltime
                    list_ot1.append(ot)
                    list_pt1.append(pt)
                    self.timeterms_response1_x.append(geop)
                    self.timeterms_response1_t.append(pt)

                self.timeterms_respLayer1 = list_pt1
                self.timeterms_response = list_pt1
                timeterms_observed = list_ot1

                if self.layer2:
                    
                    d2 = array([self.layer2[i][1] for i in range(len(self.layer2))])
                    G2 = zeros((int(len(self.layer2)),
                                   int(len(self.sources)+len(gx)+1)))

                    for i in range(len(self.layer2)):
                        
                        G2[i][self.layer2[i][-2]] = 1
                        G2[i][self.layer2[i][-1]+len(self.sources)] = 1
                        G2[i][-1] = self.layer2[i][-3]

                    sol_layer2 = solve(self.layer2, G2, d2, regw)
                    v2 = 1/sol_layer2[-1]
                    self.velocity2 = v2
                    
                    dtg2 = array([i for i in sol_layer2[len(self.sources):-1]]) #delay-time of all geophones
                    dts2 = array([i for i in sol_layer2[:len(self.sources)]]) #delay-time of all sources
                    
                    z_layer2 = gz-((dtg2*v1*v2)/(sqrt((v2**2)-(v1**2))))
                    self.z_layer2 = z_layer2
                        
                    list_ot2, list_pt2 = [],[]
                    self.timeterms_response2_x = []
                    self.timeterms_response2_t = []
                    
                    for p in self.layer2:
                        
                        s = p[-2]
                        g = p[-1]
                        x = p[-3]
                        ot = p[1] #observed traveltime
                        geop = p[0]
                        pt = dts2[s]+dtg2[g]+sol_layer2[-1]*x #predicted traveltime
                        list_ot2.append(ot)
                        list_pt2.append(pt)
                        self.timeterms_response2_x.append(geop)
                        self.timeterms_response2_t.append(pt)
                        
                    self.timeterms_respLayer2 = list_pt2
                    self.timeterms_response += list_pt2
                    timeterms_observed += list_ot2

                if self.layer3:
                    
                    d3 = array([self.layer3[i][1] for i in range(len(self.layer3))])
                    G3 = zeros((int(len(self.layer3)),
                                   int(len(self.sources)+len(gx)+1)))

                    for i in range(len(self.layer3)):
                        
                        G3[i][self.layer3[i][-2]] = 1
                        G3[i][self.layer3[i][-1]+len(self.sources)] = 1
                        G3[i][-1] = self.layer3[i][-3]

                    sol_layer3 = solve(self.layer3, G3, d3, regw)
                    v3 = 1/sol_layer3[-1] #m/s
                    self.velocity3 = v3

                    dtg3 = array([i for i in sol_layer3[len(self.sources):-1]]) #delay-time of all geophones3
                    dts3 = array([i for i in sol_layer3[:len(self.sources)]]) #delay-time of all sources3

                    if self.velocity2: upvmed = (v1+v2)/2
                    else: upvmed = v1
                        
                    z_layer3 = gz-((dtg3*upvmed*v3)/(sqrt((v3**2)-(upvmed**2))))
                    self.z_layer3 = z_layer3

                    list_ot3, list_pt3 = [],[]
                    self.timeterms_response3_x = []
                    self.timeterms_response3_t = []
                    
                    for p in self.layer3:
                        
                        s = p[-2]
                        g = p[-1]
                        geop = p[0]
                        x = p[-3]
                        ot = p[1] #observed traveltime
                        pt = dts3[s]+dtg3[g]+sol_layer3[-1]*x #predicted traveltime
                        list_ot3.append(ot)
                        list_pt3.append(pt)
                        self.timeterms_response3_x.append(geop)
                        self.timeterms_response3_t.append(pt)
                        
                    self.timeterms_respLayer3 = list_pt3
                    self.timeterms_response += list_pt3
                    timeterms_observed += list_ot3
            
                if self.layer1 and self.layer2 and not self.layer3: 

                    self.fill_layer1 = self.ax_timeterms.fill_between(gx, z_layer2, gz, color = self.layer1_color, alpha = 1,edgecolor = "k", label = "%d m/s"%v1)
                    self.fill_layer2 = self.ax_timeterms.fill_between(gx, z_layer2, min(z_layer2)*1.5, color = self.layer2_color,alpha = 1,edgecolor = "k", label = "%d m/s"%v2)
        
                elif self.layer1 and self.layer2 and self.layer3:

                    self.fill_layer1 = self.ax_timeterms.fill_between(gx, z_layer2, gz, color = self.layer1_color,alpha = 1,edgecolor = "k", label = "%d m/s"%v1)
                    self.fill_layer2 = self.ax_timeterms.fill_between(gx,z_layer2, z_layer3,color = self.layer2_color, alpha = 1,edgecolor = "k", label = "%d m/s"%v2)
                    self.fill_layer3 = self.ax_timeterms.fill_between(gx,z_layer3, min(z_layer3)*0.99, color = self.layer3_color,alpha = 1,edgecolor = "k", label = "%d m/s"%v3)

                if self.layer1 and self.layer3 and not self.layer2:

                    self.fill_layer1 = self.ax_timeterms.fill_between(gx, z_layer3, gz, color = self.layer1_color,alpha = 1,edgecolor = "k", label = "%d m/s"%v1)
                    self.fill_layer3 = self.ax_timeterms.fill_between(gx, z_layer3, min(z_layer3)*0.99, color = self.layer3_color,alpha = 1,edgecolor = "k",label = "%d m/s"%v3)
     
                self.ax_timeterms.legend(loc="best")
                self.timetermsPlot = True                
                self.fig_timeterms.canvas.draw()
                
                self.timeterms_rmse = sqrt(mean((array(self.timeterms_response)-array(timeterms_observed))**2))
                self.timeterms_relrmse = (sqrt(mean(square((array(timeterms_observed) - array(self.timeterms_response)) / array(timeterms_observed))))) * 100
                
                #messagebox.showinfo('Refrainv','Absolute RMSE = %.2f ms\nRelative RMSE = %.2f%%'%(rmse*1000,relrmse))
                self.showFit()
    
    
    def interpolateVelModelToMesh(self, mesh):
        """Interpolate the velocity model to the current mesh"""
        try:
            from scipy.interpolate import griddata
            
            # Get mesh cell centers
            cell_centers = mesh.cellCenters()
            mesh_x = np.array([cell_centers[i][0] for i in range(len(cell_centers))])
            mesh_z = np.array([cell_centers[i][1] for i in range(len(cell_centers))])
            mesh_points = np.column_stack([mesh_x, mesh_z])
            
            # Interpolate velocity values to mesh points
            interpolated_vel = griddata(
                self.velModel['points'], 
                self.velModel['values'], 
                mesh_points, 
                method='linear',
                fill_value=self.velModel['values'].mean()  # Use mean for extrapolation
            )
            
            # Convert to PyGimli RVector if needed
            if hasattr(pg, 'RVector'):
                return pg.RVector(interpolated_vel)
            else:
                return interpolated_vel
                
        except Exception as e:
            raise Exception(f"Failed to interpolate velocity model to mesh: {str(e)}")    
    def layersInterpretation(self):

        if self.data_pg:

            if self.layerInterpretationMode == False:
                
                self.layerInterpretationMode = True
                self.statusLabel.configure(text = 'Layer %d interpratation enabled!'%self.layer2interpretate)
                    
                def onpick(event):
                    
                    art = event.artist
                    artx = art.get_offsets()[0][0]
                    artt = art.get_offsets()[0][1]

                    for i in range(len(self.sources)):
                        
                        if art in self.dataArts[i][self.sources[i]]:
                            
                            arts = self.sources[i]
                            iS = i 
                            
                    if artx >= arts:
                        
                        for b in self.dataArts[iS][arts]:
                            
                            bx = b.get_offsets()[0][0]
                            iG = where(array(self.dataArts[iS][arts]) == b)[0][0]
                            
                            if arts <= bx <= artx:
                                
                                bt = b.get_offsets()[0][1]
                                
                                if self.layer2interpretate == 1 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer1:
                                    
                                    b.set_color(self.layer1_color)
                                    self.layer1.append((bx,bt,arts,abs(arts-bx),iS, iG))#geophone_position , arrival_time , source_poisition , offset , index_source , index_geophone

                                    if (bx,bt,arts,abs(arts-bx),iS, iG) in self.layer2:
                                        
                                        self.layer2.remove((bx,bt,arts,abs(arts-bx),iS, iG))
                                        
                                elif self.layer2interpretate == 2 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer1 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer2:
                                    
                                    b.set_color(self.layer2_color)
                                    self.layer2.append((bx,bt,arts,abs(arts-bx),iS, iG))
                                    
                                    if (bx,bt,arts,abs(arts-bx),iS, iG) in self.layer3:
                                        
                                        self.layer3.remove((bx,bt,arts,abs(arts-bx),iS, iG))
                                        
                                elif self.layer2interpretate == 3 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer2 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer1 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer3:

                                    b.set_color(self.layer3_color)
                                    self.layer3.append((bx,bt,arts,abs(arts-bx),iS, iG))
                                    
                    elif artx <= arts:
                        
                        for b in self.dataArts[iS][arts]:
                            
                            bx = b.get_offsets()[0][0]
                            iG = where(array(self.dataArts[iS][arts]) == b)[0][0]
                            
                            if arts >= bx >= artx:
                                
                                bt = b.get_offsets()[0][1]
                                
                                if self.layer2interpretate == 1 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer1:
                                    
                                    b.set_color(self.layer1_color)
                                    self.layer1.append((bx,bt,arts,abs(arts-bx),iS, iG))
                                    
                                    if (bx,bt,arts,abs(arts-bx),iS, iG) in self.layer2:
                                        
                                        self.layer2.remove((bx,bt,arts,abs(arts-bx),iS, iG))
                                        
                                elif self.layer2interpretate == 2 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer1 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer2:
                                    
                                    b.set_color(self.layer2_color)
                                    self.layer2.append((bx,bt,arts,abs(arts-bx),iS, iG))
                                    
                                    if (bx,bt,arts,abs(arts-bx),iS, iG) in self.layer3:
                                        
                                        self.layer3.remove((bx,bt,arts,abs(arts-bx),iS, iG))
                                        
                                elif self.layer2interpretate == 3 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer2 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer1 and (bx,bt,arts,abs(arts-bx),iS, iG) not in self.layer3:

                                    b.set_color(self.layer3_color)
                                    self.layer3.append((bx,bt,arts,abs(arts-bx),iS, iG))
                    
                    self.fig_data.canvas.draw()

                def onkey(event):
                    
                    if event.key == "1": self.layer2interpretate = 1
                    elif event.key == "2": self.layer2interpretate = 2
                    elif event.key == "3": self.layer2interpretate = 3
                    
                    self.statusLabel.configure(text = 'Layer %d interpratation enabled!'%self.layer2interpretate)
                    
                    if event.key == "C" or event.key == "c":
                        
                        if messagebox.askyesno("Refrainv", "Clear layer interpretation?"):
                            
                            del self.layer1[:]
                            del self.layer2[:]
                            del self.layer3[:]
                            
                            for i in range(len(self.sources)):
                                
                                for b in self.dataArts[i][self.sources[i]]:
                                    
                                    b.set_color("white")
                                    b.set_edgecolor("k")

                            self.fig_data.canvas.draw()

                self.timeterms_pickEvent = self.fig_data.canvas.mpl_connect('pick_event', onpick)
                self.timeterms_keyEvent = self.fig_data.canvas.mpl_connect('key_press_event', onkey)
                messagebox.showinfo('Refrapy','Layer interpretation enabled!')

            else:
                
                self.fig_data.canvas.mpl_disconnect(self.timeterms_pickEvent)
                self.fig_data.canvas.mpl_disconnect(self.timeterms_keyEvent)
                self.statusLabel.configure(text = 'Layer interpratation disabled')
                messagebox.showinfo('Refrainv','Layer interpretation disabled!')
                self.layerInterpretationMode = False
    
    def clearTomoPlot(self):

        self.fig_tomography.clf()
        self.ax_tomography = self.fig_tomography.add_subplot(111)
        self.fig_tomography.patch.set_facecolor('#F0F0F0')
        self.ax_tomography.set_title("Tomography velocity model")
        self.ax_tomography.set_xlabel("POSITION [m]")
        
        if self.z2elev: self.ax_tomography.set_ylabel("ELEVATION [m]")
        else: self.ax_tomography.set_ylabel("DEPTH [m]")

        if self.showGrid: self.ax_tomography.grid(lw = .5, alpha = .5)
        else: self.ax_tomography.grid(False)

        if self.showMerged:

            if self.layer2: self.merged_layer2.remove()
            if self.layer3: self.merged_layer3.remove()
            self.showMerged = False
        
        self.ax_tomography.set_aspect("equal")
        self.ax_tomography.spines['right'].set_visible(False)
        self.ax_tomography.spines['top'].set_visible(False)
        self.ax_tomography.yaxis.set_ticks_position('left')
        self.ax_tomography.xaxis.set_ticks_position('bottom')
        self.tomoPlot = False
        self.fig_tomography.canvas.draw()

    def clearTimeTermsPlot(self):

        self.fig_timeterms.clf()
        self.ax_timeterms = self.fig_timeterms.add_subplot(111)
        self.fig_timeterms.patch.set_facecolor('#F0F0F0')
        self.ax_timeterms.set_title("Time-terms velocity model")
        self.ax_timeterms.set_xlabel("POSITION [m]")

        if self.z2elev: self.ax_timeterms.set_ylabel("ELEVATION [m]")
        else: self.ax_timeterms.set_ylabel("DEPTH [m]")

        if self.showGrid: self.ax_timeterms.grid(lw = .5, alpha = .5)
        else: self.ax_timeterms.grid(False)

        if self.showMerged:

            if self.layer2: self.merged_layer2.remove()
            if self.layer3: self.merged_layer3.remove()
            self.showMerged = False
        
        self.ax_timeterms.set_aspect("equal")
        self.ax_timeterms.spines['right'].set_visible(False)
        self.ax_timeterms.spines['top'].set_visible(False)
        self.ax_timeterms.yaxis.set_ticks_position('left')
        self.ax_timeterms.xaxis.set_ticks_position('bottom')
        self.timetermsPlot = False
        self.fig_timeterms.canvas.draw()
                
    def runTomography(self):

        if self.data_pg:

            self.mgr = TravelTimeManager()

            tomoWindow = Toplevel(self)
            tomoWindow.title('Refrainv - Tomography')
            tomoWindow.configure(bg = "#F0F0F0")
            tomoWindow.geometry("400x800")
            tomoWindow.resizable(True,True)
            #check if on windows with nt kernel:
            if "nt" in name:
                tomoWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
            # if not, use unix formats
            else:
                tomoWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
            #tomoWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
            
            # Create main frame with scrollbar
            main_frame = Frame(tomoWindow)
            main_frame.pack(fill="both", expand=True)
            
            # Create canvas and scrollbar
            canvas = Canvas(main_frame, bg="#F0F0F0")
            scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = Frame(canvas, bg="#F0F0F0")
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Pack canvas and scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Bind mousewheel to canvas for scrolling
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            # Now use scrollable_frame instead of tomoWindow for all widgets

            def viewMesh():

                maxDepth = float(maxDepth_entry.get())
                paraDX = float(paraDX_entry.get())
                paraMaxCellSize = float(paraMaxCellSize_entry.get())
                paraQuality =float(paraQuality_entry.get())
                
                self.tomoMesh = self.mgr.createMesh(data=self.data_pg,paraDepth=maxDepth,paraDX=paraDX,paraMaxCellSize=paraMaxCellSize,quality=paraQuality)

                meshWindow = Toplevel(self)
                meshWindow.title('Refrainv - Mesh')
                meshWindow.configure(bg = "#F0F0F0")
                #meshWindow.geometry("1024x768")
                meshWindow.resizable(0,0)
                #check if on windows with nt kernel:
                if "nt" in name:
                    meshWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
                # if not, use unix formats
                else:
                    meshWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
                #meshWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())

                frame = Frame(meshWindow)
                frame.grid(row = 0, column = 0)
                fig = plt.figure(figsize = (12,6))#.2,8.62))
                fig.patch.set_facecolor('#F0F0F0')
                canvas = FigureCanvasTkAgg(fig, frame)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                canvas._tkcanvas.pack()
                
                ax = fig.add_subplot(111)
                ax.set_ylabel("POSITION [m]")
                ax.set_xlabel("DEPTH [m]")
                if self.showGrid: ax.grid(lw = .5, alpha = .5)

                #vel = ra.paraModel()
                pg.show(self.tomoMesh, ax = ax)
                #pg.viewer.mpl.drawSensors(ax, data.sensorPositions(), diam=0.5, color="k")
                ax.set_xlabel('Distance (m)')
                ax.set_ylabel('Elevation (m)')
                ax.set_title('Mesh for traveltimes tomography')

                ax.set_aspect("equal")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')

                fig.canvas.draw()
                meshWindow.tkraise()
                #print(m.dimension)
                #print(m.yMin())
                #print(m.yMax())
            def showInterpolatedVelMesh( mesh, interpolated_vel):
                """Show the interpolated velocity model on the mesh in a new window."""
                window = Toplevel(self)
                window.title('Refrainv - Interpolated Velocity Model Mesh')
                window.configure(bg="#F0F0F0")
                window.geometry("800x600")
                window.resizable(True, True)
                if "nt" in name:
                    window.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
                else:
                    window.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
            
                frame = Frame(window)
                frame.grid(row=0, column=0, sticky="NSEW")
                fig = plt.figure(figsize=(10, 8))
                fig.patch.set_facecolor('#F0F0F0')
                canvas = FigureCanvasTkAgg(fig, frame)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                canvas._tkcanvas.pack(fill="both", expand=True)
                ax = fig.add_subplot(111)
                pg.show(mesh, interpolated_vel, ax=ax, cMap=self.colormap, label="Velocity [m/s]", showMesh=True)
                ax.set_title("Interpolated Velocity Model on Mesh")
                ax.set_xlabel("Distance [m]")
                ax.set_ylabel("Depth [m]" if not self.z2elev else "Elevation [m]")
                if self.showGrid:
                    ax.grid(lw=0.5, alpha=0.5)
                ax.set_aspect("equal")
                fig.tight_layout()
                fig.canvas.draw()
                window.tkraise()    
            def showVelModel():
                """Display the loaded velocity model in a separate window"""
                if self.velModel is None:
                    messagebox.showwarning("Refrainv", "No velocity model loaded!")
                    #tomoWindow.tkraise()
                    return
                
                try:
                    # Create a new window to show the velocity model
                    velModelWindow = Toplevel(self)
                    velModelWindow.title('Refrainv - Velocity Model Preview')
                    velModelWindow.configure(bg="#F0F0F0")
                    velModelWindow.geometry("800x600")
                    velModelWindow.resizable(True, True)
                    
                    # Set icon
                    if "nt" in name:
                        velModelWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
                    else:
                        velModelWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
                    
                    # Create matplotlib figure
                    frame = Frame(velModelWindow)
                    frame.grid(row=0, column=0, sticky="NSEW")
                    
                    fig = plt.figure(figsize=(10, 8))
                    fig.patch.set_facecolor('#F0F0F0')
                    canvas = FigureCanvasTkAgg(fig, frame)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, frame)
                    toolbar.update()
                    canvas._tkcanvas.pack(fill="both", expand=True)
                    
                    # Create subplot
                    ax = fig.add_subplot(111)
                    
                    # Create contour plot
                    X, Z = np.meshgrid(self.velModel['x_coords'], self.velModel['z_coords'])
                    
                    # Create filled contour plot
                    contour = ax.contourf(X, Z, self.velModel['vel_grid'], 
                                         levels=20, cmap=self.colormap, extend="both")
                    
                    # Add colorbar
                    cbar = fig.colorbar(contour, ax=ax, label="Velocity [m/s]", format='%d')
                    
                    ax.set_title(f"Velocity Model: {path.basename(self.velModel['file_path'])}")
                    ax.set_xlabel("Distance [m]")
                    ax.set_ylabel("Depth [m]" if not self.z2elev else "Elevation [m]")
                    
                    if self.showGrid:
                        ax.grid(lw=0.5, alpha=0.5)
                    
                    ax.set_aspect("equal")
                    
                    # Add statistics
                    vel_min = float(self.velModel['values'].min())
                    vel_max = float(self.velModel['values'].max())
                    vel_mean = float(self.velModel['values'].mean())
                    
                    stats_text = (f"Grid: {self.velModel['nx']} x {self.velModel['nz']}\n"
                                 f"Min: {vel_min:.0f} m/s\n"
                                 f"Max: {vel_max:.0f} m/s\n"
                                 f"Mean: {vel_mean:.0f} m/s")
                    
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    fig.tight_layout()
                    fig.canvas.draw()
                    velModelWindow.tkraise()
                    
                except Exception as e:
                    messagebox.showerror("Refrainv", f"Error displaying velocity model: {str(e)}")
                    tomoWindow.tkraise()
            def loadVelModel():
                """Load a velocity model from a .vel file"""
                try:
                    vel_file_path = filedialog.askopenfilename(
                        title="Select Velocity Model File (.vel)",
                        initialdir=self.projPath,
                        filetypes=[("Velocity Files", "*.vel"), ("All Files", "*.*")]
                    )
                    
                    if vel_file_path:
                        # Parse the .vel file
                        with open(vel_file_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Parse header information
                        # Line 0: Format identifier (usually 'V')
                        # Line 1: Number of x nodes
                        # Line 2: Number of z nodes  
                        # Line 3: Usually 1 (parameter count)
                        # Line 4: x_min
                        # Line 5: x_max
                        # Line 6: z_min
                        # Line 7: z_max
                        # Line 8: Usually 0 (flag)
                        # Line 9: Usually 0 (flag)
                        # Line 10+: Velocity values
                        
                        nx = int(lines[1].strip())
                        nz = int(lines[2].strip())
                        x_min = float(lines[4].strip())
                        x_max = float(lines[5].strip())
                        z_min = float(lines[6].strip())
                        z_max = float(lines[7].strip())
                        
                        # Read velocity data (starting from line 11)
                        vel_data_str = ' '.join(lines[10:])
                        vel_values = [float(v) for v in vel_data_str.split()]
                        
                        # Verify we have the expected number of values
                        expected_values = nx * nz
                        if len(vel_values) != expected_values:
                            messagebox.showwarning("Refrainv", 
                                f"Warning: Expected {expected_values} velocity values, got {len(vel_values)}")
                        
                        # Create coordinate grids
                        x_coords = np.linspace(x_min, x_max, nx)
                        z_coords = np.linspace(z_min, z_max, nz)
                        
                        # Reshape velocity data to 2D grid (assuming row-major order: z varies fastest)
                        vel_grid = np.flipud(np.array(vel_values[:nx*nz]).reshape(nz, nx))*1000
                        
                        # Create mesh coordinates for interpolation
                        X, Z = np.meshgrid(x_coords, z_coords)
                        
                        # Flatten for interpolation
                        points = np.column_stack([X.ravel(), Z.ravel()])
                        values = vel_grid.ravel()
                        
                        # Store the velocity model data
                        self.velModel = {
                            'points': points,
                            'values': values,
                            'x_coords': x_coords,
                            'z_coords': z_coords,
                            'vel_grid': vel_grid,
                            'nx': nx,
                            'nz': nz,
                            'x_min': x_min,
                            'x_max': x_max,
                            'z_min': z_min,
                            'z_max': z_max,
                            'file_path': vel_file_path
                        }
                        
                        velmodel_label.config(text=f"Loaded: {path.basename(vel_file_path)}")
                        messagebox.showinfo("Refrainv", 
                            f"Velocity model loaded successfully!\n"
                            f"Grid: {nx} x {nz}\n"
                            f"X range: {x_min:.1f} - {x_max:.1f} m\n"
                            f"Z range: {z_min:.1f} - {z_max:.1f} m\n"
                            f"Velocity range: {values.min():.1f} - {values.max():.1f} m/s")
                            
                except Exception as e:
                    messagebox.showerror("Refrainv", f"Error loading velocity model: {str(e)}")
                    
                tomoWindow.tkraise()
            def clearVelModel():
                """Clear the loaded velocity model"""
                self.velModel = None
                velmodel_label.config(text="No velocity model loaded")
                messagebox.showinfo("Refrainv", "Velocity model cleared!")
                tomoWindow.tkraise()
            def loadStartModel():
                """Load a VTK start model from a previous inversion"""
                try:
                    start_model_path = filedialog.askopenfilename(
                        title="Select VTK Start Model File",
                        initialdir=self.projPath,
                        filetypes=[("VTK Files", "*.vtk"), ("All Files", "*.*")]
                    )
                    
                    if start_model_path:
                        # Load the VTK file using pygimli
                        start_mesh = pg.load(start_model_path)
                        
                        # Extract velocity data from the mesh (usually stored as 'Velocity' or 'v')
                        velocity_data = None
                        if start_mesh.haveData('Velocity'):
                            velocity_data = start_mesh['Velocity']
                        elif start_mesh.haveData('v'):
                            velocity_data = start_mesh['v']
                        elif start_mesh.haveData('slowness'):
                            velocity_data = 1.0 / start_mesh['slowness']
                        else:
                            # Try to get any data that might be velocity
                            data_keys = start_mesh.dataKeys()
                            if data_keys:
                                velocity_data = start_mesh[data_keys[0]]
                                messagebox.showwarning("Refrainv", 
                                    f"Using data field '{data_keys[0]}' as velocity. Please verify this is correct.")
                        
                        if velocity_data is not None:
                            self.startModel = velocity_data
                            self.startModelPath = start_model_path
                            startmodel_label.config(text=f"Loaded: {path.basename(start_model_path)}")
                            messagebox.showinfo("Refrainv", "Start model loaded successfully!")
                        else:
                            messagebox.showerror("Refrainv", "No velocity data found in the VTK file!")
                            
                except Exception as e:
                    messagebox.showerror("Refrainv", f"Error loading start model: {str(e)}")
                    
                tomoWindow.tkraise()
            
            def clearStartModel():
                """Clear the loaded start model"""
                self.startModel = None
                self.startModelPath = None
                startmodel_label.config(text="No start model loaded")
                messagebox.showinfo("Refrainv", "Start model cleared!")
                tomoWindow.tkraise()
            
            def showStartModel():
                """Display the loaded start model in a separate window"""
                if self.startModel is None:
                    messagebox.showwarning("Refrainv", "No start model loaded!")
                    tomoWindow.tkraise()
                    return
                
                try:
                    # Load the mesh from the VTK file
                    start_mesh = pg.load(self.startModelPath)
                    
                    # Create a new window to show the start model
                    startModelWindow = Toplevel(self)
                    startModelWindow.title('Refrainv - Start Model Preview')
                    startModelWindow.configure(bg="#F0F0F0")
                    startModelWindow.geometry("800x600")
                    startModelWindow.resizable(True, True)
                    
                    # Set icon
                    if "nt" in name:
                        startModelWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
                    else:
                        startModelWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
                    
                    # Create matplotlib figure
                    frame = Frame(startModelWindow)
                    frame.grid(row=0, column=0, sticky="NSEW")
                    
                    fig = plt.figure(figsize=(10, 8))
                    fig.patch.set_facecolor('#F0F0F0')
                    canvas = FigureCanvasTkAgg(fig, frame)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, frame)
                    toolbar.update()
                    canvas._tkcanvas.pack(fill="both", expand=True)
                    
                    # Create subplot
                    ax = fig.add_subplot(111)
                    
                    # Show the start model using pygimli
                    pg.show(start_mesh, self.startModel, ax=ax, cMap=self.colormap, 
                           label="Velocity [m/s]", showMesh=True)
                    
                    ax.set_title(f"Start Model: {path.basename(self.startModelPath)}")
                    ax.set_xlabel("Distance [m]")
                    ax.set_ylabel("Depth [m]" if not self.z2elev else "Elevation [m]")
                    
                    if self.showGrid:
                        ax.grid(lw=0.5, alpha=0.5)
                    
                    ax.set_aspect("equal")
                    
                    # Add some statistics - convert PyGimli RVector to NumPy array first
                    import numpy as np
                    vel_array = np.array(self.startModel)
                    vel_min = float(vel_array.min())
                    vel_max = float(vel_array.max())
                    vel_mean = float(vel_array.mean())
                    
                    stats_text = f"Min: {vel_min:.0f} m/s\nMax: {vel_max:.0f} m/s\nMean: {vel_mean:.0f} m/s\nCells: {len(self.startModel)}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    fig.tight_layout()
                    fig.canvas.draw()
                    startModelWindow.tkraise()
                    
                except Exception as e:
                    messagebox.showerror("Refrainv", f"Error displaying start model: {str(e)}")
                    tomoWindow.tkraise()

            def runInversion():
                if self.tomoPlot:
                    self.clearTomoPlot()
                
                # Notify user about GPU usage
                if self.gpu_enabled:
                    messagebox.showinfo("Refrainv", "GPU acceleration is enabled.\n\nThe inversion will use CUDA for accelerated computations.")
                
                start_timing = datetime.now()    
                
                maxDepth = float(maxDepth_entry.get())
                paraDX = float(paraDX_entry.get())
                paraMaxCellSize = float(paraMaxCellSize_entry.get())
                paraQuality = float(paraQuality_entry.get())
                self.tomoMesh = self.mgr.createMesh(data=self.data_pg, paraDepth=maxDepth, 
                                                   paraDX=paraDX, paraMaxCellSize=paraMaxCellSize, 
                                                   quality=paraQuality)
            
                lam = float(lam_entry.get())
                zWeigh = float(zWeigh_entry.get())
                vTop = float(vTop_entry.get())
                vBottom = float(vBottom_entry.get())
                minVelLimit = float(minVelLimit_entry.get())
                maxVelLimit = float(maxVelLimit_entry.get())
                self.minVelLimit = minVelLimit
                self.maxVelLimit = maxVelLimit
                secNodes = int(secNodes_entry.get())
                maxIter = int(maxIter_entry.get())
                
                xngrid = xngrid_entry.get()
                yngrid = yngrid_entry.get()
                nlevels = nlevels_entry.get()
                
                # Prepare inversion parameters
                invert_kwargs = {
                    'data': self.data_pg,
                    'mesh': self.tomoMesh,
                    'verbose': True,
                    'lam': lam,
                    'zWeight': zWeigh,
                    'useGradient': True,
                    'vTop': vTop,
                    'vBottom': vBottom,
                    'maxIter': maxIter,
                    'limits': [minVelLimit, maxVelLimit],
                    'secNodes': secNodes
                }
                
                # Check for start models (VTK or .vel file)
                start_model_used = None
                
                # Priority 1: VTK start model (existing functionality)
                if self.startModel is not None:
                    try:
                        if len(self.startModel) != self.tomoMesh.cellCount():
                            messagebox.showwarning("Refrainv", 
                                "VTK start model mesh size differs from current mesh. Interpolating...")
                            start_model_interp = pg.interpolate(self.startModel, 
                                                               srcMesh=pg.load(self.startModelPath), 
                                                               destMesh=self.tomoMesh)
                            invert_kwargs['startModel'] = start_model_interp
                        else:
                            invert_kwargs['startModel'] = self.startModel
                        start_model_used = f"VTK: {path.basename(self.startModelPath)}"
                    except Exception as e:
                        messagebox.showwarning("Refrainv", 
                            f"Could not use VTK start model: {str(e)}. Checking for velocity model...")
                
                # Priority 2: .vel file start model (new functionality)
                if 'startModel' not in invert_kwargs and hasattr(self, 'velModel') and self.velModel is not None:
                    try:
                        interpolated_vel = self.interpolateVelModelToMesh(self.tomoMesh)
                        invert_kwargs['startModel'] = 1./interpolated_vel # 1/ because of slowness
                        start_model_used = f"VEL: {path.basename(self.velModel['file_path'])}"
                        messagebox.showinfo("Refrainv", "Using .vel file as start model for inversion.")
                        showInterpolatedVelMesh(mesh=self.tomoMesh, interpolated_vel=interpolated_vel)
                        
                    except Exception as e:
                        messagebox.showwarning("Refrainv", 
                            f"Could not use .vel start model: {str(e)}. Proceeding without start model.")
                
                if start_model_used:
                    messagebox.showinfo("Refrainv", f"Using start model: {start_model_used}")
                
                vest = self.mgr.invert(**invert_kwargs)
                         
                # Keep entered options and add start model info
                self.tomostandards["depth"] = maxDepth
                self.tomostandards["dx"] = paraDX
                self.tomostandards["cellseize"] = paraMaxCellSize
                self.tomostandards["quality"] = paraQuality
                self.tomostandards["lamda"] = lam
                self.tomostandards["zweight"] = zWeigh
                self.tomostandards["vtop"] = vTop
                self.tomostandards["vbottom"] = vBottom
                self.tomostandards["minvel"] = minVelLimit
                self.tomostandards["maxvel"] = maxVelLimit
                self.tomostandards["secnodes"] = secNodes
                self.tomostandards["maxiter"] = maxIter
                self.tomostandards["gridx"] = xngrid
                self.tomostandards["gridy"] = yngrid
                self.tomostandards["nlevels"] = nlevels
                
                self.parameters_tomo = [maxDepth, paraDX, paraMaxCellSize, lam, zWeigh, vTop, vBottom, 
                                       minVelLimit, maxVelLimit, secNodes, maxIter,
                                       int(xngrid_entry.get()), int(yngrid_entry.get()), int(nlevels_entry.get()),
                                       self.mgr.inv.maxIter, self.mgr.inv.relrms(), self.mgr.inv.chi2(), 
                                       self.mgr.inv.inv.iter()]
                
                end_timing = datetime.now()
                self.tomotiming = end_timing - start_timing
                
                # Regular pygimli save
                self.mgr.saveResult(self.projPath)
                
                plotContourModel()
                if self.showRayPath: 
                    self.RayPaths = self.mgr.drawRayPaths(self.ax_tomography, color=self.rayPathColor)    
            def plotContourModel():
   
                xzvcs = column_stack((self.mgr.paraDomain.cellCenters(),
                                     self.mgr.model,
                                     self.mgr.coverage(),
                                     self.mgr.standardizedCoverage()))
                x = (xzvcs[:,0])
                z = (xzvcs[:,1])
                v = (xzvcs[:,3])
                c = (xzvcs[:,4])
                s = (xzvcs[:,5])

                self.tomoModel_x = x
                self.tomoModel_z = z
                self.tomoModel_v = v
                self.tomoModel_c = c
                self.tomoModel_s = s
                
                nx = int(xngrid_entry.get())
                ny = int(yngrid_entry.get())
                x_grid = linspace(min(x), max(x), nx)
                y_grid = linspace(min(z), max(z), ny)
                xi,zi = meshgrid(x_grid,y_grid)
                
                # Use GPU-accelerated griddata if GPU is enabled
                vi = accelerate_griddata((x, z), v, (xi, zi), method='linear', use_gpu=self.gpu_enabled)

                nlevels = int(nlevels_entry.get())
                
                cm = self.ax_tomography.contourf(xi, zi, vi, levels=nlevels,
                                    cmap=self.colormap, extend="both")

                self.cmPlot = cm

                divider = make_axes_locatable(self.ax_tomography)
                cax = divider.append_axes("right", size="2%", pad=0.05)

                cbar = self.fig_tomography.colorbar(cm,orientation="vertical", label = "[m/s]",
                             format='%d',cax=cax)

                x2max = [max(self.sgx)]
                x2min = [min(self.sgx)]
                
                for i in range(len(self.sources)): x2max.append(max(self.xdata[i])); x2min.append(min(self.xdata[i]))
                
                xlim = sorted(self.sgx)+[max(x2max),min(x2min)]
                zlim = self.sgz+[self.tomoMesh.yMin(),self.tomoMesh.yMin()]

                self.topographyx, self.topographyz = [], []
                

                for i in range(len(self.sgx)):

                    if self.sgx[i] <= max(x2max) and self.sgx[i] >= min(x2min): self.topographyx.append(self.sgx[i]); self.topographyz.append(self.sgz[i])

                xblank, zblank = [], []

                for i in range(len(xlim)):

                    if xlim[i] <= max(x2max) and xlim[i] >= min(x2min):

                        if xlim[i] not in xblank: xblank.append(xlim[i]); zblank.append(zlim[i])

                xblank = xlim
                zblank = zlim

                print(xblank)
                print(zblank)
                
                self.xbln = xblank
                self.zbln = zblank

                self.ax_tomography.plot(self.topographyx, self.topographyz, c= "k", lw = 1.5)
                
                limits = [(i,j) for i,j in zip(xblank,zblank)]
                
                clippath = Path(limits)
                patch = PathPatch(clippath, facecolor='none', alpha = 0)
                self.ax_tomography.add_patch(patch)

                patch = PathPatch(clippath, facecolor='none', alpha = 0)
                self.ax_tomography.add_patch(patch)

                for c in cm.collections: c.set_clip_path(patch)

                if self.showRayPath: self.RayPaths=self.mgr.drawRayPaths(self.ax_tomography,color=self.rayPathColor)

                if self.showSources: self.sourcesPlot_tomography = self.ax_tomography.scatter(self.sx,self.sz, marker="*",c="y",edgecolor="k",s=self.dx*20,zorder=99)

                if self.showGeophones: self.geophonesPlot_tomography = self.ax_tomography.scatter(self.gx,self.gz, marker=7,c="k",s=self.dx*10,zorder=99)
                    
                self.ax_tomography.set_xlim(min(x2min),max(x2max))
                self.fig_tomography.canvas.draw()
                self.tomoPlot = True
                
                self.showFit()
                tomoWindow.destroy()

            offsets = []
            
            for i in range(len(self.sources)):

                for x in self.xdata[i][self.sources[i]]:

                    offsets.append(abs(self.sources[i]-x))
            #set standards, if conditions in case we are in second run during execution
            # if not hasattr(self,"tomostandards"):
                # self.tomostandards={"depth":str(max(offsets)/3),
                               # "dx":"0.33",
                               # "cellseize":str(3*(self.gx[1]-self.gx[0])),
                               # "quality":"32",
                               # "lamda":"100",
                               # "zweight":"0.2",
                               # "vtop":"300",
                               # "vbottom":"3000",
                               # "minvel":"100",
                               # "maxvel":"4000",
                               # "secnodes":"3",
                               # "maxiter":"10",
                               # "gridx":"1000",
                               # "gridy":"1000",
                               # "nlevels":"20"}
            # #load from config
            if not hasattr(self,"tomostandards"):
            
                try:
                    config_path=self.projPath+"/"+"config.json"
                    with open(config_path, "r") as f:
                        self.tomostandards = json.load(f)
                except Exception as e:
                    print(f"Failed to load tomo standards config: {e}")
                    self.tomostandards = {
                    "depth": str(max(offsets) / 3),
                    "dx": "0.33",
                    "cellseize": str(3 * (self.gx[1] - self.gx[0])),
                    "quality": "32",
                    "lamda": "100",
                    "zweight": "0.2",
                    "vtop": "300",
                    "vbottom": "3000",
                    "minvel": "100",
                    "maxvel": "4000",
                    "secnodes": "3",
                    "maxiter": "10",
                    "gridx": "1000",
                    "gridy": "1000",
                    "nlevels": "20"
                                        }
            
            # --- Mesh Options Section ---
            mesh_section = Label(scrollable_frame, text="Mesh Options", font=("Arial", 12, "bold"), bg="#F0F0F0")
            mesh_section.grid(row=0, column=0, columnspan=2, pady=(15, 5), sticky="EW")

            Label(scrollable_frame, text="Maximum depth (max offset = %.2f m)" % max(offsets), bg="#F0F0F0").grid(row=1, column=0, pady=3, sticky="E")
            maxDepth_entry = Entry(scrollable_frame, width=10)
            maxDepth_entry.grid(row=1, column=1, pady=3, sticky="W")
            maxDepth_entry.insert(0, self.tomostandards["depth"])
            maxDepth_entry.config(highlightbackground="#228B22", highlightcolor="#228B22", highlightthickness=1)

            Label(scrollable_frame, text="Node every # times receiver distance:", bg="#F0F0F0").grid(row=2, column=0, pady=3, sticky="E")
            paraDX_entry = Entry(scrollable_frame, width=10)
            paraDX_entry.grid(row=2, column=1, pady=3, sticky="W")
            paraDX_entry.insert(0, self.tomostandards["dx"])

            Label(scrollable_frame, text="Maximum cell size", bg="#F0F0F0").grid(row=3, column=0, pady=3, sticky="E")
            paraMaxCellSize_entry = Entry(scrollable_frame, width=10)
            paraMaxCellSize_entry.grid(row=3, column=1, pady=3, sticky="W")
            paraMaxCellSize_entry.insert(0, self.tomostandards["cellseize"])

            Label(scrollable_frame, text="Quality parameter (higher = finer mesh)", bg="#F0F0F0").grid(row=4, column=0, pady=3, sticky="E")
            paraQuality_entry = Entry(scrollable_frame, width=10)
            paraQuality_entry.grid(row=4, column=1, pady=3, sticky="W")
            paraQuality_entry.insert(0, self.tomostandards["quality"])

            Button(scrollable_frame, text="View mesh", command=viewMesh, bg="#e0e0e0").grid(row=5, column=0, columnspan=2, pady=(5, 10), sticky="EW")

            # --- Inversion Options Section ---
            inv_section = Label(scrollable_frame, text="Inversion Options", font=("Arial", 12, "bold"), bg="#F0F0F0")
            inv_section.grid(row=6, column=0, columnspan=2, pady=(10, 5), sticky="EW")

            Label(scrollable_frame, text="Smoothing (lam)", bg="#F0F0F0").grid(row=7, column=0, pady=3, sticky="E")
            lam_entry = Entry(scrollable_frame, width=10)
            lam_entry.grid(row=7, column=1, pady=3, sticky="W")
            lam_entry.insert(0, self.tomostandards["lamda"])
            lam_entry.bind("<Enter>", lambda e: lam_entry.config(bg="#e6f7ff"))
            lam_entry.bind("<Leave>", lambda e: lam_entry.config(bg="white"))

            Label(scrollable_frame, text="Vertical/horizontal smoothing (zweight)", bg="#F0F0F0").grid(row=8, column=0, pady=3, sticky="E")
            zWeigh_entry = Entry(scrollable_frame, width=10)
            zWeigh_entry.grid(row=8, column=1, pady=3, sticky="W")
            zWeigh_entry.insert(0, self.tomostandards["zweight"])

            Label(scrollable_frame, text="Velocity at the top of the model", bg="#F0F0F0").grid(row=9, column=0, pady=3, sticky="E")
            vTop_entry = Entry(scrollable_frame, width=10)
            vTop_entry.grid(row=9, column=1, pady=3, sticky="W")
            vTop_entry.insert(0, self.tomostandards["vtop"])

            Label(scrollable_frame, text="Velocity at the bottom of the model", bg="#F0F0F0").grid(row=10, column=0, pady=3, sticky="E")
            vBottom_entry = Entry(scrollable_frame, width=10)
            vBottom_entry.grid(row=10, column=1, pady=3, sticky="W")
            vBottom_entry.insert(0, self.tomostandards["vbottom"])

            Label(scrollable_frame, text="Minimum velocity limit", bg="#F0F0F0").grid(row=11, column=0, pady=3, sticky="E")
            minVelLimit_entry = Entry(scrollable_frame, width=10)
            minVelLimit_entry.grid(row=11, column=1, pady=3, sticky="W")
            minVelLimit_entry.insert(0, self.tomostandards["minvel"])

            Label(scrollable_frame, text="Maximum velocity limit", bg="#F0F0F0").grid(row=12, column=0, pady=3, sticky="E")
            maxVelLimit_entry = Entry(scrollable_frame, width=10)
            maxVelLimit_entry.grid(row=12, column=1, pady=3, sticky="W")
            maxVelLimit_entry.insert(0, self.tomostandards["maxvel"])

            Label(scrollable_frame, text="# of secondary nodes", bg="#F0F0F0").grid(row=13, column=0, pady=3, sticky="E")
            secNodes_entry = Entry(scrollable_frame, width=10)
            secNodes_entry.grid(row=13, column=1, pady=3, sticky="W")
            secNodes_entry.insert(0, self.tomostandards["secnodes"])

            Label(scrollable_frame, text="Maximum # of iterations", bg="#F0F0F0").grid(row=14, column=0, pady=3, sticky="E")
            maxIter_entry = Entry(scrollable_frame, width=10)
            maxIter_entry.grid(row=14, column=1, pady=3, sticky="W")
            maxIter_entry.insert(0, self.tomostandards["maxiter"])

            # --- Start Model Section ---
            start_section = Label(scrollable_frame, text="Start Model", font=("Arial", 12, "bold"), bg="#F0F0F0")
            start_section.grid(row=15, column=0, columnspan=2, pady=(15, 5), sticky="EW")

            startmodel_label = Label(scrollable_frame, text="No start model loaded", fg="gray", bg="#F0F0F0")
            startmodel_label.grid(row=16, column=0, columnspan=2, pady=2, sticky="EW")

            Button(scrollable_frame, text="Load VTK Start Model", command=loadStartModel, bg="#e0e0e0").grid(row=17, column=0, pady=2, sticky="EW")
            Button(scrollable_frame, text="Clear Start Model", command=clearStartModel, bg="#e0e0e0").grid(row=17, column=1, pady=2, sticky="EW")
            Button(scrollable_frame, text="Show Start Model", command=showStartModel, bg="#e0e0e0").grid(row=18, column=0, columnspan=2, pady=2, sticky="EW")

            # --- Velocity Model (.vel) Section ---
            vel_section = Label(scrollable_frame, text="Velocity Model (.vel file)", font=("Arial", 12, "bold"), bg="#F0F0F0")
            vel_section.grid(row=19, column=0, columnspan=2, pady=(15, 5), sticky="EW")

            velmodel_label = Label(scrollable_frame, text="No velocity model loaded", fg="gray", bg="#F0F0F0")
            velmodel_label.grid(row=20, column=0, columnspan=2, pady=2, sticky="EW")

            Button(scrollable_frame, text="Load .vel Model", command=loadVelModel, bg="#e0e0e0").grid(row=21, column=0, pady=2, sticky="EW")
            Button(scrollable_frame, text="Clear .vel Model", command=clearVelModel, bg="#e0e0e0").grid(row=21, column=1, pady=2, sticky="EW")
            Button(scrollable_frame, text="Show .vel Model", command=showVelModel, bg="#e0e0e0").grid(row=22, column=0, columnspan=2, pady=2, sticky="EW")

            # --- Contour Plot Options Section ---
            contour_section = Label(scrollable_frame, text="Contour Plot Options", font=("Arial", 12, "bold"), bg="#F0F0F0")
            contour_section.grid(row=23, column=0, columnspan=2, pady=(15, 5), sticky="EW")

            Label(scrollable_frame, text="# of nodes for gridding (x)", bg="#F0F0F0").grid(row=24, column=0, pady=3, sticky="E")
            xngrid_entry = Entry(scrollable_frame, width=10)
            xngrid_entry.grid(row=24, column=1, pady=3, sticky="W")
            xngrid_entry.insert(0, self.tomostandards["gridx"])

            Label(scrollable_frame, text="# of nodes for gridding (y)", bg="#F0F0F0").grid(row=25, column=0, pady=3, sticky="E")
            yngrid_entry = Entry(scrollable_frame, width=10)
            yngrid_entry.grid(row=25, column=1, pady=3, sticky="W")
            yngrid_entry.insert(0, self.tomostandards["gridy"])

            Label(scrollable_frame, text="# of contour levels", bg="#F0F0F0").grid(row=26, column=0, pady=3, sticky="E")
            nlevels_entry = Entry(scrollable_frame, width=10)
            nlevels_entry.grid(row=26, column=1, pady=3, sticky="W")
            nlevels_entry.insert(0, self.tomostandards["nlevels"])

            # --- Action Buttons Section ---
            Button(scrollable_frame, text="Run Inversion", command=runInversion, bg="#228B22", fg="white", font=("Arial", 11, "bold")).grid(row=27, column=0, columnspan=2, pady=(15, 5), sticky="EW")
            Button(scrollable_frame, text="Batch Inversion", command=self.batchTomography, bg="#0055aa", fg="white", font=("Arial", 11, "bold")).grid(row=28, column=0, columnspan=2, pady=5, sticky="EW")

            # --- Info Section ---
            info_text = (
                "Tips:\n"
                "- Use 'View mesh' to preview before running inversion.\n"
                "- Start model (.vtk) or velocity model (.vel) can be used as initial guess.\n"
                "- Batch inversion lets you explore parameter ranges automatically."
            )
            Label(scrollable_frame, text=info_text, font=("Arial", 9), fg="#444", bg="#F0F0F0", justify="left", wraplength=340).grid(row=29, column=0, columnspan=2, pady=(10, 5), sticky="EW")

            tomoWindow.tkraise()

    def saveResults(self):

        if self.tomoPlot:
              
            now=datetime.now()
            nowstring=now.strftime("%Y%m%d-%H%M%S")
            os.mkdir(self.projPath+"/models/"+nowstring+"/")
            velwithdummy=self.tomoModel_s*self.tomoModel_v
            for i,sv in enumerate(velwithdummy):
                if self.tomoModel_c[i]==np.inf*-1: 
                    velwithdummy[i]=0
            savetxt(self.projPath+"/models/"+nowstring+"/%s_xzv.txt"%(self.lineName),c_[self.tomoModel_x,self.tomoModel_z,self.tomoModel_v, self.tomoModel_c, self.tomoModel_s,velwithdummy], fmt = "%.2f", header = "x z velocity converage standardized coverage sc_v",comments="")
            self.fig_tomoFit.savefig(self.projPath+"/models/"+nowstring+"/%s_tomography_response.jpeg"%(self.lineName), format="jpeg",dpi = 300,transparent=True)
            self.fig_tomography.savefig(self.projPath+"/models/"+nowstring+"/%s_tomography_model.jpeg"%(self.lineName), format="jpeg",dpi = 300,transparent=True)
            savetxt(self.projPath+"/models/"+nowstring+"/%s_topography.txt"%(self.lineName),c_[self.topographyx,self.topographyz], fmt = "%.2f", header = "x z",comments="")
            savetxt(self.projPath+"/models/"+nowstring+"/%s_tomography_limits.bln"%(self.lineName),c_[self.xbln,self.zbln], fmt = "%.2f", header = "%d,1"%len(self.xbln),comments="")
          
            #get the paths:
                

           
                  
            if hasattr(self, 'RayPaths'):
                pathcount=len(self.RayPaths.get_paths())
                
                # Export individual raypaths
                with open(self.projPath+"/models/"+nowstring+"/%s_raypaths.bln"%(self.lineName),mode='w+') as f:
                    for i in range(pathcount):
                       patharray=self.RayPaths.get_paths()[i].vertices
                       n=len(patharray)
                       f.write(str(n)+',-1\n')
                       pd.DataFrame(patharray).to_csv(path_or_buf=f,sep=',',header=False,index=False,line_terminator='\n')
                       f.write('\n')
                
                # Generate and export raypath hull
                try:
                    # Collect all raypath points
                    all_points = []
                    buffer=10
                    for i in range(pathcount):
                        patharray = self.RayPaths.get_paths()[i].vertices
                        all_points.extend(patharray)
                    
                    if len(all_points) >= 3:  # ConvexHull needs at least 3 points
                        all_points = np.array(all_points)
                        
                        # Generate convex hull
                        hull = ConvexHull(all_points)
                        hull_points = all_points[hull.vertices]
                        #first point is the upperrightmost one
                        #create the points of the "outside" poly:
                        #upper_rightest=(hull_points[0][0]+2*buffer,hull_points[0][1]+2*buffer)
                        #upper_2ndrightest=(hull_points[0][0]+buffer,hull_points[0][1]+buffer)
                        #upper_lefttest=(hull_points[1][0]-2*buffer,hull_points[0][1]+2*buffer)
                        #upper_2ndlefttest=(hull_points[1][0]-buffer,hull_points[0][1]+2*buffer)
                        
                        #lower_left=(hull_points[1][0]-2*buffer,self.tomoModel_z.min-buffer)
                        #lower_right=(hull_points[0][0]+2*buffer,self.tomoModel_z.min-buffer)
                        # Export hull as BLN file (Surfer compatible format)
                        with open(self.projPath+"/models/"+nowstring+"/%s_raypaths_hull.bln"%(self.lineName), 'w') as f:
                            # For Surfer BLN format: number of points (including closing point), flag (1 for closed polygon)
                            n_hull_points = len(hull_points) + 1  # +1 for closing the polygon
                            f.write(f"{n_hull_points}, 1\n")  # Space after comma for better compatibility
                            
                            # Write all hull points
                            for point in hull_points:
                                f.write(f"{point[0]:.6f}, {point[1]:.6f}\n")  # Space after comma
                            
                            # Close the polygon by adding the first point at the end
                            f.write(f"{hull_points[0][0]:.6f}, {hull_points[0][1]:.6f}\n")
                            
                        print(f"Raypath hull exported with {n_hull_points} points")
                    else:
                        print("Not enough points to generate raypath hull")
                        
                except Exception as e:
                    print(f"Error generating raypath hull: {e}")

                       
                    
            
            if self.tomography_3d_ready: savetxt(self.projPath+"/models/%s_tomography_xyzv.txt"%(self.lineName),c_[self.new_x_tomography,self.new_y_tomography,self.tomoModel_z,self.tomoModel_v], fmt = "%.2f", header = "x y z velocity",comments="")

            with open(self.projPath+"/models/"+nowstring+"/%s_tomography_parameters.txt"%(self.lineName),"w") as outFile:

                outFile.write("%s - Traveltimes tomography parameters\n\n"%self.lineName)
                outFile.write("Mesh options\n")
                outFile.write("Maximum depth %.2f\n"%(self.parameters_tomo[0]))
                outFile.write("Nodes created every # times receiver-distance  %.2f\n"%(self.parameters_tomo[1]))
                outFile.write("Maximum cell size %.2f\n\n"%(self.parameters_tomo[2]))
                outFile.write("Inversion options\n")
                outFile.write("Smoothing (lam) %.2f\n"%(self.parameters_tomo[3]))
                outFile.write("Vertical to horizontal smoothing (zweigh) %.2f\n"%(self.parameters_tomo[4]))
                outFile.write("Velocity at the top of the model %.2f\n"%(self.parameters_tomo[5]))
                outFile.write("Velocity at the bottom of the model %.2f\n"%(self.parameters_tomo[6]))
                outFile.write("Minimum velocity limit %.2f\n"%(self.parameters_tomo[7]))
                outFile.write("Maximum velocity limit %.2f\n"%(self.parameters_tomo[8]))
                outFile.write("# of secondary nodes %d\n"%(self.parameters_tomo[9]))
                outFile.write("Maximum # of iterations %d\n"%(self.parameters_tomo[10]))
                outFile.write("Actually done # of iterations %d\n\n"%(self.parameters_tomo[17]))
                outFile.write("Contour plot options\n")
                outFile.write("# of nodes for gridding (x) %d\n"%(self.parameters_tomo[11]))
                outFile.write("# of nodes for gridding (y) %d\n"%(self.parameters_tomo[12]))
                outFile.write("# of contour levels %d\n\n"%(self.parameters_tomo[13]))
                outFile.write("Model response\n")
                outFile.write("Final iteration %d\n"%(self.parameters_tomo[17]))
                outFile.write("Relative RMSE %.2f\n"%(self.parameters_tomo[15]))
                outFile.write("Chi² %.2f\n"%(self.parameters_tomo[16]))
                outFile.write("Calculation Time: "+ str(self.tomotiming))

        if self.timetermsPlot:

            if self.layer1:

                savetxt(self.projPath+"/models/"+nowstring+"/%s_timeterms_layer1.txt"%(self.lineName),c_[self.gx_timeterms,self.gz_timeterms], fmt = "%.2f", header = "x z",comments="")
                
                if self.timeterms_3d_ready: savetxt(self.projPath+"/models/"+nowstring+"/%s_timeterms_layer1_xyz.txt"%(self.lineName),c_[self.new_x_timeterms,self.new_y_timeterms,self.gz_timeterms], fmt = "%.2f", header = "x y z",comments="")
                
            if self.layer2:

                savetxt(self.projPath+"/models/"+nowstring+"/%s_timeterms_layer2.txt"%(self.lineName),c_[self.gx_timeterms,self.z_layer2], fmt = "%.2f", header = "x z",comments="")

                if self.timeterms_3d_ready: savetxt(self.projPath+"/models/"+nowstring+"/%s_timeterms_layer2_xyz.txt"%(self.lineName),c_[self.new_x_timeterms,self.new_y_timeterms,self.z_layer2], fmt = "%.2f", header = "x y z",comments="")
                    
            if self.layer3:

                savetxt(self.projPath+"/models/"+nowstring+"/%s_timeterms_layer3.txt"%(self.lineName),c_[self.gx_timeterms,self.z_layer3], fmt = "%.2f", header = "x z",comments="")

                if self.timeterms_3d_ready: savetxt(self.projPath+"/models/"+nowstring+"/%s_timeterms_layer3_xyz.txt"%(self.lineName),c_[self.new_x_timeterms,self.new_y_timeterms,self.z_layer3], fmt = "%.2f", header = "x y z",comments="")
            
            self.fig_data.savefig(self.projPath+"/models/"+nowstring+"/%s_layers_assignment.jpeg"%(self.lineName), format="jpeg",dpi = 300,transparent=True)
            self.fig_timetermsFit.savefig(self.projPath+"/models/"+nowstring+"/%s_timeterms_response.jpeg"%(self.lineName), format="jpeg",dpi = 300,transparent=True)
            self.fig_timeterms.savefig(self.projPath+"/models/"+nowstring+"/%s_timeterms_model.jpeg"%(self.lineName), format="jpeg",dpi = 300,transparent=True)

        if self.timetermsPlot or self.tomoPlot: messagebox.showinfo(title="Refrainv", message="All results saved in %s"%(self.projPath+"/models/"))
    
    def showFit(self):

        if self.data_pg:

            fitWindow = Toplevel(self)
            fitWindow.title('Refrainv - Fit')
            fitWindow.configure(bg = "#F0F0F0")
            fitWindow.resizable(0,0)
            #check if on windows with nt kernel:
            if "nt" in name:
                fitWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
            # if not, use unix formats
            else:
                fitWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
          #  fitWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())

            frame1 = Frame(fitWindow)
            frame1.grid(row = 0, column = 0)
            fig1 = plt.figure(figsize = (7.1,8.62))
            fig1.patch.set_facecolor('#F0F0F0')
            canvas1 = FigureCanvasTkAgg(fig1, frame1)
            canvas1.draw()
            toolbar1 = NavigationToolbar2Tk(canvas1, frame1)
            toolbar1.update()
            canvas1._tkcanvas.pack()

            frame2 = Frame(fitWindow)
            frame2.grid(row = 0, column = 1)
            fig2 = plt.figure(figsize = (7.1,8.62))
            fig2.patch.set_facecolor('#F0F0F0')
            canvas2 = FigureCanvasTkAgg(fig2, frame2)
            canvas2.draw()
            toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
            toolbar2.update()
            canvas2._tkcanvas.pack()
            
            ax_fitTimeterms = fig1.add_subplot(111)
            ax_fitTimeterms.set_ylabel("TRAVELTIME [s]")
            ax_fitTimeterms.set_xlabel("POSITION [m]")
            if self.showGrid: ax_fitTimeterms.grid(lw = .5, alpha = .5)
            ax_fitTimeterms.set_title("Time-terms inversion fit")

            ax_fitTomography = fig2.add_subplot(111)

            if self.data_pg:

                pg.physics.traveltime.drawFirstPicks(ax_fitTimeterms, self.data_pg, marker="o", lw = 1)
                pg.physics.traveltime.drawFirstPicks(ax_fitTomography, self.data_pg, marker="o", lw = 1)
                ax_fitTimeterms.set_title("Observed traveltimes (time-terms panel)")
                ax_fitTomography.set_title("Observed traveltimes (tomography panel)")
                ax_fitTimeterms.set_ylabel("TRAVELTIME [s]")
                ax_fitTimeterms.set_xlabel("POSITION [m]")
                if self.showGrid: ax_fitTimeterms.grid(lw = .5, alpha = .5)
                ax_fitTomography.set_ylabel("TRAVELTIME [s]")
                ax_fitTomography.set_xlabel("POSITION [m]")
                if self.showGrid: ax_fitTomography.grid(lw = .5, alpha = .5)
            
            if self.timetermsPlot:
            
                if self.timeterms_response1_t: ax_fitTimeterms.scatter(self.timeterms_response1_x,self.timeterms_response1_t,marker="x",c="r",s=self.dx*10,zorder=99)
                if self.timeterms_response2_t: ax_fitTimeterms.scatter(self.timeterms_response2_x,self.timeterms_response2_t,marker="x",c="r",s=self.dx*10,zorder=99)
                if self.timeterms_response3_t: ax_fitTimeterms.scatter(self.timeterms_response3_x,self.timeterms_response3_t,marker="x",c="r",s=self.dx*10,zorder=99)
                ax_fitTimeterms.set_title("Time-terms model response\nRRMSE = %.2f%%"%self.timeterms_relrmse) #mgr.absrms() mgr.chi2()
            
            if self.tomoPlot:
                
                ax_fitTomography.set_title("Tomography model response\n%d iterations | RRMSE = %.2f%% | Chi^2 = %.2f%%"%(self.mgr.inv.inv.iter(),self.mgr.inv.relrms(),self.mgr.inv.chi2())) #mgr.absrms() mgr.chi2()
                pg.physics.traveltime.drawFirstPicks(ax_fitTomography, self.data_pg, marker="o", lw = 0)
                #pg.physics.traveltime.drawFirstPicks(ax_fitTomography, self.data_pg, tt= self.mgr.inv.response, marker="", linestyle = "--")
                ax_fitTomography.scatter(self.gx,self.mgr.inv.response,marker="x",c="r",zorder=99,s=self.dx*10)
                ax_fitTomography.invert_yaxis()

            legend_elements = [Line2D([0], [0], marker='o', color='k', label='Observed data', markerfacecolor='k', markersize=self.dx),
                               Line2D([0], [0], marker='x',lw=0, color='r', label='Model response', markerfacecolor='r', markersize=self.dx)]
            ax_fitTimeterms.legend(handles=legend_elements, loc='best')
            ax_fitTomography.legend(handles=legend_elements, loc='best')

            for art in ax_fitTimeterms.get_lines(): art.set_color("k")
            for art in ax_fitTomography.get_lines(): art.set_color("k")    
            
            ax_fitTimeterms.invert_yaxis()
            ax_fitTomography.invert_yaxis()
            
            fig1.canvas.draw()
            fig2.canvas.draw()
            self.fig_tomoFit = fig2
            self.fig_timetermsFit = fig1
            fitWindow.tkraise()

    def showPgResult(self):

        if self.tomoPlot:

            pgWindow = Toplevel(self)
            pgWindow.title('Refrainv - Velocity model with mesh')
            pgWindow.configure(bg = "#F0F0F0")
            pgWindow.resizable(0,0)
            #check if on windows with nt kernel:
            if "nt" in name:
                pgWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
            # if not, use unix formats
            else:
                pgWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
            #pgWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())

            frame = Frame(pgWindow)
            frame.grid(row = 0, column = 0)
            fig = plt.figure(figsize = (14.2,8.62))
            fig.patch.set_facecolor('#F0F0F0')
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas._tkcanvas.pack()
            ax_pg = fig.add_subplot(111)
            
            pg.show(self.tomoMesh, self.mgr.model, label = "[m/s]",
                    cMin=self.minVelLimit,cMax=self.maxVelLimit,cMap=self.colormap,ax = ax_pg)

            if self.showRayPath: self.mgr.drawRayPaths(ax = ax_pg,color=self.rayPathColor)

            ax_pg.set_ylabel("DEPTH [m]")
            ax_pg.set_xlabel("DISTANCE [m]")
            if self.showGrid: ax_pg.grid(lw = .5, alpha = .5)
            ax_pg.set_title("Tomography velocity model")
            
            fig.canvas.draw()
            pgWindow.tkraise()

    def build3d(self):

        if self.tomoPlot or self.timetermsPlot:

            if not self.coords_3d:

                messagebox.showinfo(title="Refrainv", message="Select now the file containing the survey line coordinates (4-column file: distance,x,y,elevation)")

                file_3d = filedialog.askopenfilename(title='Open', initialdir = self.projPath+"/gps/", filetypes=[('Text file', '*.txt'),('CSV file', '*.csv')])
                d,x,y,z = [],[],[],[]
                
                with open(file_3d, "r") as file:

                    lines = file.readlines()

                    for l in lines:

                        dist = l.replace(' ', ',').replace('	',',').replace(';',',').replace('\n','').split(',')[0]
                        xcoord = l.replace(' ', ',').replace('	',',').replace(';',',').replace('\n','').split(',')[1]
                        ycoord = l.replace(' ', ',').replace('	',',').replace(';',',').replace('\n','').split(',')[2]
                        elev = l.replace(' ', ',').replace('	',',').replace(';',',').replace('\n','').split(',')[3]
                        d.append(float(dist))
                        x.append(float(xcoord))
                        y.append(float(ycoord))
                        z.append(float(elev))

                self.coords_3d.append(d)
                self.coords_3d.append(x)
                self.coords_3d.append(y)
                self.coords_3d.append(z)

            if self.coords_3d:

                def save3dtomo():

                    if self.tomoPlot:

                        savetxt(self.projPath+"/models/%s_tomography_xyzv.txt"%(self.lineName),c_[self.new_x_tomography,self.new_y_tomography,self.tomoModel_z,self.tomoModel_v], fmt = "%.2f", header = "x y z velocity",comments="")
                        messagebox.showinfo(title="Refrainv", message="File saved in %s"%(self.projPath+"/models/"))
                        plot3dwindow.tkraise()

                def save3dtimeterms():

                    if self.timetermsPlot:
                        
                        if self.layer1: savetxt(self.projPath+"/models/%s_timeterms_layer1_xyz.txt"%(self.lineName),c_[self.new_x_timeterms,self.new_y_timeterms,self.gz_timeterms], fmt = "%.2f", header = "x y z",comments="")
                        if self.layer2: savetxt(self.projPath+"/models/%s_timeterms_layer2_xyz.txt"%(self.lineName),c_[self.new_x_timeterms,self.new_y_timeterms,self.z_layer2], fmt = "%.2f", header = "x y z",comments="")
                        if self.layer3: savetxt(self.projPath+"/models/%s_timeterms_layer3_xyz.txt"%(self.lineName),c_[self.new_x_timeterms,self.new_y_timeterms,self.z_layer3], fmt = "%.2f", header = "x y z",comments="")
                        messagebox.showinfo(title="Refrainv", message="Files saved in %s"%(self.projPath+"/models/"))
                        plot3dwindow.tkraise()

                plot3dwindow = Toplevel(self)
                plot3dwindow.title('Refrainv - 3D view')
                plot3dwindow.configure(bg = "#F0F0F0")
                plot3dwindow.geometry("1600x900")
                plot3dwindow.resizable(0,0)
                #check if on windows with nt kernel:
                if "nt" in name:
                    plot3dWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
                # if not, use unix formats
                else:
                    plot3dWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
               # plot3dwindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())

                frame_buttons = Frame(plot3dwindow)
                frame_buttons.grid(row = 0, column = 0, columnspan=100,sticky="W")

                Button(frame_buttons,text="Save time-terms model 3D file",command=save3dtimeterms).grid(row=0,column=0,sticky="W")
                Button(frame_buttons,text="Save tomography model 3D file",command=save3dtomo).grid(row=0,column=1,sticky="W")
                
                frame1 = Frame(plot3dwindow)
                frame1.grid(row = 1, column = 0, rowspan=2)
                frame2 = Frame(plot3dwindow)
                frame2.grid(row = 1, column = 1)
                frame3 = Frame(plot3dwindow)
                frame3.grid(row = 2, column = 1)
                
                fig1 = plt.figure(figsize = (5,5))
                fig1.patch.set_facecolor('#F0F0F0')
                canvas1 = FigureCanvasTkAgg(fig1, frame1)
                canvas1.draw()
                toolbar1 = NavigationToolbar2Tk(canvas1, frame1)
                toolbar1.update()
                canvas1._tkcanvas.pack()

                fig2 = plt.figure(figsize = (11.1,3.95))
                fig2.patch.set_facecolor('#F0F0F0')
                canvas2 = FigureCanvasTkAgg(fig2, frame2)
                canvas2.draw()
                toolbar2 = NavigationToolbar2Tk(canvas2, frame2)
                toolbar2.update()
                canvas2._tkcanvas.pack()

                fig3 = plt.figure(figsize = (11.1,3.95))
                fig3.patch.set_facecolor('#F0F0F0')
                canvas3 = FigureCanvasTkAgg(fig3, frame3)
                canvas3.draw()
                toolbar3 = NavigationToolbar2Tk(canvas3, frame3)
                toolbar3.update()
                canvas3._tkcanvas.pack()

                ax_coords = fig1.add_subplot(111, aspect = "equal")
                ax_coords.set_ylabel("Y [m]")
                ax_coords.set_xlabel("X [m]")
                if self.showGrid:ax_coords.grid(lw = .5, alpha = .5)
                ax_coords.set_title("Survey coordinates")
                ax_coords.set_facecolor('#F0F0F0')
                
                ax_3d_timeterms = fig2.add_subplot(111, projection = "3d")
                ax_3d_timeterms.set_box_aspect((1, 1, 1))

                ax_3d_tomo = fig3.add_subplot(111, projection = "3d")
                ax_3d_tomo.set_box_aspect((1, 1, 1))

                ax_3d_timeterms.set_ylabel("Y [m]")
                ax_3d_timeterms.set_xlabel("X [m]")
                ax_3d_timeterms.set_zlabel("ELEVATION [m]")
                if self.showGrid:ax_3d_timeterms.grid(lw = .5, alpha = .5)
                ax_3d_timeterms.set_title("Time-terms velocity model")
                ax_3d_timeterms.set_facecolor('#F0F0F0')
                
                ax_3d_tomo.set_ylabel("Y [m]")
                ax_3d_tomo.set_xlabel("X [m]")
                ax_3d_tomo.set_zlabel("ELEVATION [m]")
                if self.showGrid:ax_3d_tomo.grid(lw = .5, alpha = .5)
                ax_3d_tomo.set_title("Tomography velocity model")
                ax_3d_tomo.set_facecolor('#F0F0F0')

                ax_coords.ticklabel_format(useOffset=False, style='plain')
                ax_3d_timeterms.ticklabel_format(useOffset=False, style='plain')
                ax_3d_tomo.ticklabel_format(useOffset=False, style='plain')
                
                fig1.canvas.draw()
                fig2.canvas.draw()
                fig3.canvas.draw()

                if self.timetermsPlot:
                    
                    fx = interp1d(self.coords_3d[0],self.coords_3d[1], kind = "linear", fill_value = "extrapolate")
                    fy = interp1d(self.coords_3d[0],self.coords_3d[2], kind = "linear", fill_value = "extrapolate")
                    self.new_x_timeterms = fx(self.gx_timeterms)
                    self.new_y_timeterms = fy(self.gx_timeterms)
                    ax_coords.plot(self.coords_3d[1],self.coords_3d[2],c="k")
                    
                    if self.layer1: ax_3d_timeterms.plot(self.new_x_timeterms,self.new_y_timeterms,self.gz_timeterms,c = self.layer1_color)
                    if self.layer2: ax_3d_timeterms.plot(self.new_x_timeterms,self.new_y_timeterms,self.z_layer2,c = self.layer2_color)
                    if self.layer3: ax_3d_timeterms.plot(self.new_x_timeterms,self.new_y_timeterms,self.z_layer3,c = self.layer3_color)

                    self.timeterms_3d_ready = True

                if self.tomoPlot:

                    fx = interp1d(self.coords_3d[0],self.coords_3d[1], kind = "linear", fill_value = "extrapolate")
                    fy = interp1d(self.coords_3d[0],self.coords_3d[2], kind = "linear", fill_value = "extrapolate")
                    self.new_x_tomography = fx(self.tomoModel_x)
                    self.new_y_tomography = fy(self.tomoModel_x)
                    ax_coords.plot(self.coords_3d[1],self.coords_3d[2],c="k")
                    cm = ax_3d_tomo.scatter(self.new_x_tomography,self.new_y_tomography,self.tomoModel_z,c = self.tomoModel_v, cmap = self.colormap, s = self.dx)
                    self.tomography_3d_ready = True
                
                fig1.canvas.draw()
                fig2.canvas.draw()
                fig3.canvas.draw()
                plot3dwindow.tkraise()
    
    def plotOptions(self):

        def rayPath():
            
            if self.showRayPath == False:

                show = messagebox.askyesno("Refrainv", "Do you want to show the ray path?")

                if show:

                    self.showRayPath = True

                    if self.tomoPlot:

                        self.RayPaths=self.mgr.drawRayPaths(self.ax_tomography,color=self.rayPathColor)
                        self.fig_tomography.canvas.draw()

                    messagebox.showinfo(title="Refrainv", message="The ray path view has been enabled!")
                    plotOptionsWindow.tkraise()

            else:

                hide = messagebox.askyesno("Refrainv", "Do you want to hide the ray path?")

                if hide:

                    self.showRayPath = False

                    if self.tomoPlot:
                        
                        for art in self.ax_tomography.collections:

                            if str(type(art)) == "<class 'matplotlib.collections.LineCollection'>": art.remove()

                        self.fig_tomography.canvas.draw()

                    messagebox.showinfo(title="Refrainv", message="The ray path view has be disabled!")
                    plotOptionsWindow.tkraise()

        def rayPathLineColor():

            new_color = simpledialog.askstring("Refrainv","Enter the new ray path line color (must be accepted by matplotlib):")

            if is_color_like(new_color):

                self.rayPathColor = new_color
                
                if self.tomoPlot:

                    messagebox.showinfo(title="Refrainv", message="To update the ray path color you must now hide and show the ray path!")
                    self.showRayPath = True
                    rayPath() #force removal of ray paths
                    rayPath() #plot it
                    
                messagebox.showinfo(title="Refrainv", message="The ray path line color has been changed")
                plotOptionsWindow.tkraise()

            else: messagebox.showerror(title="Refrainv", message="Invalid color!"); plotOptionsWindow.tkraise()
            
        def colormap():

            new_cmap = simpledialog.askstring("Refrainv","Enter the new color map to be used (must be accepted by matplotlib):")

            if new_cmap in plt.colormaps():

                self.colormap = new_cmap

                if self.tomoPlot:

                    self.cmPlot.set_cmap(self.colormap)
                    self.fig_tomography.canvas.draw()

                messagebox.showinfo(title="Refrainv", message="The color map has been changed!")
                plotOptionsWindow.tkraise()

            else: messagebox.showerror(title="Refrainv", message="Invalid color map!"); plotOptionsWindow.tkraise()

        def layer1_color():

            new_color = simpledialog.askstring("Refrainv","Enter the new layer 1 color (must be accepted by matplotlib):")

            if is_color_like(new_color):

                self.layer1_color = new_color
                
                if self.timetermsPlot:

                    self.fill_layer1.set_color(self.layer1_color)
                    self.ax_timeterms.legend(loc="best")
                    self.fig_timeterms.canvas.draw()

                messagebox.showinfo(title="Refrainv", message="The layer 1 color has been changed!")
                plotOptionsWindow.tkraise()

            else: messagebox.showerror(title="Refrainv", message="Invalid color!"); plotOptionsWindow.tkraise()

        def layer2_color():

            new_color = simpledialog.askstring("Refrainv","Enter the new layer 2 color (must be accepted by matplotlib):")

            if is_color_like(new_color):

                self.layer2_color = new_color
                
                if self.timetermsPlot:

                    self.fill_layer2.set_color(self.layer2_color)
                    self.ax_timeterms.legend(loc="best")
                    self.fig_timeterms.canvas.draw()

                messagebox.showinfo(title="Refrainv", message="The layer 2 color has been changed!")
                plotOptionsWindow.tkraise()

            else: messagebox.showerror(title="Refrainv", message="Invalid color!"); plotOptionsWindow.tkraise()

        def layer3_color():

            new_color = simpledialog.askstring("Refrainv","Enter the new layer 3 color (must be accepted by matplotlib):")

            if is_color_like(new_color):

                self.layer3_color = new_color
                
                if self.timetermsPlot:

                    self.fill_layer3.set_color(self.layer3_color)
                    self.ax_timeterms.legend(loc="best")
                    self.fig_timeterms.canvas.draw()
                    
                messagebox.showinfo(title="Refrainv", message="The layer 3 color has been changed!")
                plotOptionsWindow.tkraise()

            else: messagebox.showerror(title="Refrainv", message="Invalid color!"); plotOptionsWindow.tkraise()

        def dataLinesColor():

            new_color = simpledialog.askstring("Refrainv","Enter the new traveltimes line color (must be accepted by matplotlib):")

            if is_color_like(new_color):

                self.data_color = new_color
                
                if self.data_pg:

                    for line in self.dataLines: line.set_color(self.data_color)
                    
                    self.fig_data.canvas.draw()
                    
                messagebox.showinfo(title="Refrainv", message="The traveltimes line color has been changed!")
                plotOptionsWindow.tkraise()

            else: messagebox.showerror(title="Refrainv", message="Invalid color!"); plotOptionsWindow.tkraise()

        def geophonePosition():

            if self.showGeophones == False:

                show = messagebox.askyesno("Refrainv", "Show receivers location on velocity models?")
    
                if show:

                    self.showGeophones = True
                    
                    if self.timetermsPlot:

                        self.geophonesPlot_timeterms = self.ax_timeterms.scatter(self.gx,self.gz, marker=7,c="k",s=self.dx*10,zorder=99)
                        self.fig_timeterms.canvas.draw()

                    if self.tomoPlot:

                        self.geophonesPlot_tomography = self.ax_tomography.scatter(self.gx,self.gz, marker=7,c="k",s=self.dx*10,zorder=99)
                        self.fig_tomography.canvas.draw()

                    messagebox.showinfo(title="Refrainv", message="Receivers location will be displayed on velocity models!")
                    plotOptionsWindow.tkraise()

            else:
            
                hide = messagebox.askyesno("Refrainv", "Hide receivers location on velocity models?")
    
                if hide:

                    self.showGeophones = False
                    
                    if self.geophonesPlot_timeterms: self.geophonesPlot_timeterms.remove(); self.fig_timeterms.canvas.draw()
                    if self.geophonesPlot_tomography: self.geophonesPlot_tomography.remove(); self.fig_tomography.canvas.draw()

                    messagebox.showinfo(title="Refrainv", message="Receivers location removed from velocity models!")
                    plotOptionsWindow.tkraise()

        def sourcePosition():

            if self.showSources == False:

                show = messagebox.askyesno("Refrainv", "Show sources location?")
    
                if show:

                    self.showSources = True

                    if self.data_pg:

                        self.sourcesPlot_data = self.ax_data.scatter(self.sx,array(self.sx)*0, marker="*",c="y",edgecolor="k",s=self.dx*20,zorder=99)
                        self.fig_data.canvas.draw()
                    
                    if self.timetermsPlot:

                        self.sourcesPlot_timeterms = self.ax_timeterms.scatter(self.sx,self.sz, marker="*",c="y",edgecolor="k",s=self.dx*20,zorder=99)
                        self.fig_timeterms.canvas.draw()

                    if self.tomoPlot:

                        self.sourcesPlot_tomography = self.ax_tomography.scatter(self.sx,self.sz, marker="*",c="y",edgecolor="k",s=self.dx*20,zorder=99)
                        self.fig_tomography.canvas.draw()

                    messagebox.showinfo(title="Refrainv", message="Sources location will be displayed!")
                    plotOptionsWindow.tkraise()

            else:
            
                hide = messagebox.askyesno("Refrainv", "Hide sources location?")
    
                if hide:

                    self.showSources = False
                    
                    if self.sourcesPlot_data: self.sourcesPlot_data.remove(); self.fig_data.canvas.draw()
                    if self.sourcesPlot_timeterms: self.sourcesPlot_timeterms.remove(); self.fig_timeterms.canvas.draw()
                    if self.sourcesPlot_tomography: self.sourcesPlot_tomography.remove(); self.fig_tomography.canvas.draw()

                    messagebox.showinfo(title="Refrainv", message="Sources location removed from velocity models!")
                    plotOptionsWindow.tkraise()

        def gridLines():
    
            if self.projReady:

                if self.showGrid == False:
                
                    show = messagebox.askyesno("Refrainv", "Show grid lines?")
        
                    if show:

                        self.showGrid = True
                        self.ax_data.grid(lw = .5, alpha = .5)
                        self.ax_timeterms.grid(lw = .5, alpha = .5)
                        self.ax_tomography.grid(lw = .5, alpha = .5)
                        self.fig_data.canvas.draw()
                        self.fig_timeterms.canvas.draw()
                        self.fig_tomography.canvas.draw()
                        messagebox.showinfo(title="Refrainv", message="Grid lines have been enabled!")
                        plotOptionsWindow.tkraise()

                else:

                    hide = messagebox.askyesno("Refrainv", "Hide grid lines?")
        
                    if hide:

                        self.showGrid = False
                        self.ax_data.grid(False)
                        self.ax_timeterms.grid(False)
                        self.ax_tomography.grid(False)
                        self.fig_data.canvas.draw()
                        self.fig_timeterms.canvas.draw()
                        self.fig_tomography.canvas.draw()
                        messagebox.showinfo(title="Refrainv", message="Grid lines have been disabled!")
                        plotOptionsWindow.tkraise()
        
        plotOptionsWindow = Toplevel(self)
        plotOptionsWindow.title('Refrainv - Plot options')
        plotOptionsWindow.configure(bg = "#F0F0F0")
        plotOptionsWindow.geometry("350x450")
        plotOptionsWindow.resizable(0,0)
        #check if on windows with nt kernel:
        if "nt" in name:
            plotOptionsWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
        # if not, use unix formats
        else:
            plotOptionsWindow.iconbitmap("@"+getcwd()+"/images/ico_refrapy.xbm")
       # plotOptionsWindow.iconbitmap("%s/images/ico_refrapy.ico"%getcwd())
        Label(plotOptionsWindow, text = "Plot options",font=("Arial", 11)).grid(row=0,column=0,sticky="EW",pady=5,padx=65)
        Button(plotOptionsWindow,text="Show/hide ray path", command = rayPath, width = 30).grid(row = 1, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Change ray path line color", command = rayPathLineColor, width = 30).grid(row = 2, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Change colormap", command = colormap, width = 30).grid(row = 3, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Change layer 1 color", command = layer1_color, width = 30).grid(row = 4, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Change layer 2 color", command = layer2_color, width = 30).grid(row = 5, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Change layer 3 color", command = layer3_color, width = 30).grid(row = 6, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Show/hide receiver positions", command = geophonePosition, width = 30).grid(row = 7, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Show/hide source positions", command = sourcePosition, width = 30).grid(row = 8, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Show/hide grid lines", command = gridLines, width = 30).grid(row = 9, column = 0,pady=5,padx=65)
        Button(plotOptionsWindow,text="Change traveltimes line color", command = dataLinesColor, width = 30).grid(row = 10, column = 0,pady=5,padx=65)
        
        plotOptionsWindow.tkraise()
        
app = Refrainv()
app.mainloop()
