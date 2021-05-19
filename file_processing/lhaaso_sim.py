# -*- coding: utf-8 -*-
"""
Detector grid simulator

Object-oriented detector array mask for event reading.

"""

# utils
import numpy as np
import pandas as pd
import collections as cll
import multiprocessing as mp
from event import *

#plot
import matplotlib.pyplot as plt
# import seaborn as sns




"""
 DETECTOR
 detector unit, saves event history

 params
 x,y position of the center of the detector  (m)
 l deterctor size  (m)
 E_th Energy threshold for detection  (GeV)
 detectable array with the NAMES of the detectable particles
"""
class detector:
    def __init__(self,x,y,l,E_th,detectable):
        self.x = x
        self.y = y
        self.x0 = x-l/2
        self.y0 = y-l/2
        self.x1 = x+l/2
        self.y1 = y+l/2
        self.l = l
        self.detectable = detectable;
        self.history=[];
        self.E_th = E_th;
        self.detected = 0;


    # if event in the range of the detector, saves event in history
    # and returns true. otherwise, returns false.
    def inRange(self,event):
        if(event["x"]>self.x0 and event["x"]<self.x1 and event["y"]>self.y0 and event["y"]<self.y1):
            if(event["energy"] >= self.E_th and event["particle"] in self.detectable):
                self.history.append(event)
                return True
        return False


    # resets detector history
    def blank(self):
        self.history =[]
        self.detected=0;




"""
# DETECTOR_LINE
defines a finite line of detectors along
x coordinate (y constant). uniform separation
between detectors

params
 y y-coordinate of the line (m)
 xo starting point in x coordinate (m)
 long length of the line  (m)
 sep separetion between detectors (m)
 E_th Energy threshold for detection (GeV)
 detectable array with the NAMES of the detectable particles
 ID idenification name
"""
class detector_line:
    def __init__(self,y,xo,long,sep,phase,l,detectable,E_th):
        self.y = y
        self.xo = xo
        self.xf = xo+long

        self.detectable = detectable
        self.E_th = E_th

        self.detectors = []


        x_act = xo+phase
        while(x_act<=self.xf):
            d = detector(x_act,y,l,E_th,detectable)
            self.detectors.append(d)
            x_act = x_act + sep


    # if event in the range of one of the detectors in the line,
    #saves event in history and returns true. otherwise, returns false.
    def inRange(self,event):
        registered = False
        if(len(self.detectors)>0):

            xs = np.argmin([np.abs(d.x-event["x"]) for d in self.detectors])
            d = self.detectors[xs]
            if(d.inRange(event)):
                registered=True
                return registered

        return registered


    def give_history(self):
        history = []
        for d in self.detectors:
            for ev in d.history:
                history.append(ev)
        return history

    def blank(self):
        for d in self.detectors:
            d.blank()



"""
# DETECTOR_TRIANGULAR_GRID
# Constructs a ciruclar area covered with detectors aranged in a triangular grid
# params
# xo,yo cooridnates of the center of the circle (m)
# ro radius of the circle (m)
# s separation between lines (m)[separation between detectors in the same line  is
#  automatically constructed to form equilateral triangle grid with sep = s * 2/sqrt(3)]
# l size of the individual detector (m)
# E_th Energy threshold for detection (GeV)
# detectable array with the NAMES of the detectable particles
"""
class detector_triangular_grid:
    def __init__(self,xo,yo,ro,s,l,detectable,E_th,name="unnamed_det_grid"):
        self.detectable = detectable
        self.E_th = E_th
        self.name = name
        self.detector_lines = []
        self.detected = 0
        y_act = yo-ro
        count = 0
        n_det = 0
        while(y_act<=yo+ro):

            # triangular grid
            if(count%2==0):
                dl = detector_line(y_act,xo-ro,2*ro,2*s/np.sqrt(3),0,l,detectable,E_th)
            else:
                dl = detector_line(y_act,xo-ro,2*ro,2*s/np.sqrt(3),s/np.sqrt(3),l,detectable,E_th)

            #circular geometry
            dl.detectors = [d for d in dl.detectors if np.sqrt((d.x-xo)**2 + (d.y-yo)**2)<=ro]

            if(len(dl.detectors)>0):
                for d in dl.detectors:
                    n_det+=1

                self.detector_lines.append(dl)
            y_act= y_act+s
            count = count +1
        print("DETECTOR GRID BUILT : {} \n {} Detector lines \n {} Detectors \n Energy threshold {} GeV".format(self.name,len(self.detector_lines),len(self.give_detectors()),self.E_th))
        print(" detects ",[p for p in self.detectable])

    # returns a list with the detector objects
    def give_detectors(self):
        det = []
        for dl in self.detector_lines:
            for d in dl.detectors:
                det.append(d)
        return det


    # returns a list with the history of detected particles
    def give_history(self):
        history = []
        for dl in self.detector_lines:
            for ev in dl.give_history():
                ev["detected"]=self.name
                history.append(ev)
        return history




        # process a single event
        # return 1 if particle is within detection conditions, else 0
    def  proc_single(self,event):
        registered = 0
        selected_idx = np.argmin(np.array([np.abs(dl.y-event["y"]) for dl in self.detector_lines]))
        d = self.detector_lines[selected_idx]
        if(d.inRange(event)):
            registered=1;
        self.detected = self.detected + registered
        return registered



    def process_events(self,events):
        # pre-selects the events that are above a defined energy threshold and
        # are among the detectable particles
        events = list(events.T.to_dict().values())
        useful_evs = [ev for ev in events if np.logical_and(ev["energy"]>=self.E_th,ev["particle"] in self.detectable)]
        print(len(useful_evs)," particles within energy and type conditions")

        # counter for detected particles
        det = 0
        for ev in useful_evs:
            det += self.proc_single(ev) #process a single event, soring in detector memory

        return det

    # delete  memory of detected particles
    def blank(self):
        self.detected = 0;
        for d in self.detector_lines:
            d.blank()





    def plot_sketch(self,e_color,f_color="none"):
        for dl in self.detector_lines:
           for d in dl.detectors:
               rect = plt.Rectangle((d.x0,d.y0),d.l,d.l,facecolor=f_color,edgecolor=e_color)
               plt.gca().add_patch(rect)

    def plot_event(self,color,ref_value=10):
        for dl in self.detector_lines:
            for d in dl.detectors:
                n_events = len(d.history)
                n = n_events/ref_value if n_events<ref_value else 1
                color_fc = (1,1,1) if n_events==0 else (1-n,1-n,1-n)
                color_ec = (0.9,0.9,0.9) if n_events==0 else color
                rect = plt.Rectangle((d.x0,d.y0),d.l,d.l,facecolor=color_fc,edgecolor=color_ec)
                plt.gca().add_patch(rect)




# FUNCTIONS
"""
AUXILIAR FUNCTIONS
"""
# receives a list of detector grids and particle data.
#returns a list of particle objects detected by the grids
def evaluate_events(detectors,particles):
    evs = []
    for det in detectors: # for each detector grid
        n = det.process_events(particles)  # n counts of particles in this detector
        e = det.give_history() # particle list in the detector's history
        for ev in e: # for each particle
            ev["detector"] = det.name # set an attribute accounting detection
            evs.append(ev)
        print("{}: {} particles".format(det.name,n))
    return evs



# Generates  the Lhaaso grid detectors and process the particles provided.
# returns a dataframe with the particles detected
def lhaaso_evaluate_events(evs):


    #evaluates the events through the detector mask
    dets = build_lhaaso()
    ev_list = evaluate_events(dets,evs)

    # returns a dataframe with the particle list remaining
    return pd.DataFrame(ev_list)


def build_lhaaso():
    #CREATE SCINTILLATOR GRID FOR EM PARTICLE DETECTION
    sc_radius = 575
    sc_sep = 15*np.sqrt(3)/2
    sc_size = 1
    sc_detectable = ["photon"]
    sc_E_th = 0.003 #3 Mev
    scintillator_grid = detector_triangular_grid(0,0,sc_radius,sc_sep,sc_size,sc_detectable,sc_E_th,name="em_scintillator_array")

    #CREATE MUON DETECTOR GRID FOR HADRON AND MUON DETECTION
    mu_radius = 575
    mu_sep = 30*np.sqrt(3)/2
    mu_size = 7
    mu_detectable = ["proton","neutron","mu+","mu-"]
    mu_E_th = 1.3 #1.3 Gev
    muon_grid = detector_triangular_grid(7,0,mu_radius,mu_sep,mu_size,mu_detectable,mu_E_th,name="muon_detector_array")

    return [scintillator_grid,muon_grid]



# Builds summary dataframe with the number of counts per detector per shower, energy, time and position data.
def build_final_df(ev_list,labels=None):

    shower_summary = []
    detector_counts = ev_list.groupby(["event","detector"]).size()

    for n_shower in ev_list["event"].unique():

        # measured values
        tot_en = ev_list[ev_list["event"]==n_shower]["energy"].sum() # Total energy detected (in GeV)
        time_disp = ev_list[ev_list["event"]==n_shower]["time"].std() # std time (in nanoseconds)
        space_disp = ev_list[ev_list["event"] == n_shower]["x"].std()*ev_list[ev_list["event"] == n_shower]["y"].std() #in (m2)

        # shower object
        actual_shower = {"shower":n_shower,
        "dt":time_disp, # time dispersion
        "dxy":space_disp, # spatial dispersionç
        "E":tot_en} #total energy in detectors

        # for each detector grid, append the total count to the shower object
        for det_name in ev_list["detector"].unique():
            try:
                actual_shower[det_name]=detector_counts[(n_shower,det_name)]
                if(labels):
                    actual_shower["origin"] = labels
                #append shower object to summary
                shower_summary.append(actual_shower)
            except:
                pass

    return pd.DataFrame(shower_summary).fillna(0).drop_duplicates(["shower"]).astype("int32").sort_values("shower").reset_index().drop("index",axis=1)
