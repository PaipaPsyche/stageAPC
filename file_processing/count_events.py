# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:55:45 2021

@author: david
"""
#!pip install corsikaio

from corsikaio import CorsikaParticleFile
import event
import lhaaso_sim
import numpy as np
# import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd


dictID = { # name , charge, mass(GeV)
    1:["photon",0,0],
    2:["e+",1,0.000511],
    3:["e-",-1,0.000511],
    5:["mu+",1,0.105],
    6:["mu-",-1,0.105],
    13:["neutron",0,0.939],
    14:["proton",1,0.938]
}

def main():
    in_dir = "in/"
    out_dir = "out/"

    # sc_radius = 575
    # sc_sep = 15*np.sqrt(3)/2
    # sc_size = 1
    # sc_detectable = ["photon"]
    # sc_E_th = 0.003 #3 Mev
    # scintillator_grid = lhaaso_sim.detector_triangular_grid(0,0,sc_radius,sc_sep,sc_size,sc_detectable,sc_E_th,name="em_scintillator_array")
    #
    # mu_radius = 575
    # mu_sep = 30*np.sqrt(3)/2
    # mu_size = 7
    # mu_detectable = ["proton","neutron","mu+","mu-"]
    # mu_E_th = 1.3 #1.3 Gev
    # muon_grid = lhaaso_sim.detector_triangular_grid(7,0,mu_radius,mu_sep,mu_size,mu_detectable,mu_E_th,name="muon_detector_array")
    #



    #LOAD photon shower
    in_file = in_dir+'DAT000001'
    f_ph=CorsikaParticleFile(in_file)
    #Energy Thresshold in Gev
    events_ph = lhaaso_sim.file_to_events(f_ph,dictID,E_th = 0.003)#,gen__pckl=out_dir+"ph250.pkl")
    f_ph.close()
    print("created events_ph")

    lhaaso_sim.event_summary(events_ph)

    #LOAD proton shower
    in_file = in_dir+'DAT000002'
    f_pr=CorsikaParticleFile(in_file)
    events_pr =  lhaaso_sim.file_to_events(f_pr,dictID,E_th=1.3)#,gen__pckl=out_dir+"pr250.pkl")
    f_pr.close()
    print("created events_pr")

    lhaaso_sim.event_summary(events_pr)

    # APPLYING MASK


    e_ph = lhaaso_sim.lhaaso_evaluate_events(events_ph)
    fn_ph = lhaaso_sim.build_final_df(e_ph,1)
    fn_ph.to_csv(out_dir+"final_ph.csv")


    e_pr = lhaaso_sim.lhaaso_evaluate_events(events_pr)
    fn_pr = lhaaso_sim.build_final_df(ev_pr,14)
    fn_pr.to_csv(out_dir+"final_pr.csv")

main()
