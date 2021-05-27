# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:51:34 2021

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

    # for each file asociate the content information
    # [corsika  output file , output csv name, coriska  ID of prim particle]


    files = [["DAT000001","200ph50Tev.csv",1],
            ["DAT000002","200pr50Tev.csv",14],
            ["DAT000003","200ph100Tev.csv",1],
            ["DAT000004","200pr100Tev.csv",14],
            ["DAT000005","200ph150Tev.csv",1],
            ["DAT000006","200pr150Tev.csv",14]]


    for tuple_n in files:
        #LOAD photon shower
        in_file = in_dir+tuple_n[0]
        f_ph=CorsikaParticleFile(in_file)
        #Energy Thresshold in Gev
        eth = 0.003
        if(tuple_n[2]==14):
            eth = 1.3
        events_ph = lhaaso_sim.file_to_events(f_ph,dictID,E_th = eth)#,gen__pckl=out_dir+"ph250.pkl")
        f_ph.close()
        print("file {} extracted into {} events".format(tuple_n[0],len(events_ph)))

        # APPLYING MASK


        e_ph = lhaaso_sim.lhaaso_evaluate_events(events_ph)
        fn_ph = lhaaso_sim.build_final(e_ph,tuple_n[2])
        fn_ph.to_csv(out_dir+tuple_n[1])


main()
