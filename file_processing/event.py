# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:11:29 2021

@author: david

Event managing
"""
# utils
import numpy as np
import pandas as pd
import collections as cll
import pickle as pkl




# import multiprocessing as mp
# def run_multiprocessing(func, i, n_processors):
#     with mp.Pool(processes=n_processors) as pool:
#         return pool.map(func, i)




#returns the azimuthal angle for the
# vector in coordinates x,y
def assign_phi(x,y):
    phi = 0
    if(x>0):
        if(y>0):
            phi = np.arctan(y/x)
        else:
            phi = 2*np.pi + np.arctan(y/x)
    elif(x==0):
        phi = (np.pi/2)*np.sign(y)
    elif(x<0):
        phi = np.pi + np.arctan(y/x)
    return phi%(2*np.pi)


def event_summary(events):

    for shower in events["event"].unique():
        print("shower ",shower)


        part_ev = [ev["particle"] for _,ev in events.iterrows() if ev["event"]==shower]
        print(cll.Counter(part_ev))



# CONVERTING CORSIKA FILES TO PARTICLE DATA TABLES (DATAFRAME)
# takes as parameter a file f (coriska DATXXXXX) and
# return the particle events with ID numbers existing in dictID
#params
# df   return dataframe (return dict list if false)
# max_sghower  maximum number of showers to process
# gen__pckl  generate pkl file with particle data (binary file)
def file_to_events(f,dictID,E_th=None,df=True,max_shower=None,gen__pckl=False):


    p_obj = [] #saving particles as objects (dictionaries)

    #counters
    tot_par = 0 #total particles
    shower_count = 0 #total particle showers


    # for each shower
    for e in f:
        # if maximum number of showers is set
        if (max_shower and shower_count >= max_shower):
            break

        shower_count+=1

        typp=e.particles['particle_description']
        # particle_description = 1000*ID + 10* Hadr. generation + no. of obs. level
        typ=np.trunc(np.array(typp)/1000.) # particle ID

        # momentum in GeV
        px=e.particles['px']
        py=e.particles['py']
        pz=e.particles['pz']

        #XY position when hitting ground in cm from common reference point
        x=e.particles['x']/1.e2 #cm to meters
        y=e.particles['y']/1.e2 #cm to meters

        #time when hitting ground in ns
        t=e.particles['t']


        #mass in GeV

        for i in range(len(typ)):
            code = typ[i] # particle ID
            if(code in dictID.keys()): # if ID in dictID

                particle = dictID[code] #store code of origin particle

                # E2 = p2 + m2
                E = np.sqrt(px[i]**2+py[i]**2+pz[i]**2+particle[2]**2)
                if(E_th and E<E_th):
                    pass


                particle_object = {
                    "event":shower_count,
                    "particle": particle[0],
                    # "charge":particle[1],
                    # "mass":particle[2], #in GeV/c2
                     "energy":E, #in GeV
                    # "log_energy":np.log10(E),
                    # "px":px[i], # in GeV/c
                    # "py":py[i],
                    # "pz":pz[i],
                    # "momentum":np.sqrt(px[i]**2+py[i]**2+pz[i]**2),
                    # "phi":assign_phi(px[i],py[i]),
                    "x":x[i],# meters
                    "y":y[i], #meters
                    "time":t[i] #*(10**(-9)) #nanoseconds to seconds
                }

                p_obj.append(particle_object)
            tot_par+=1
    print("Total particles: {}".format(tot_par))
    print("Total showers: {}".format(shower_count))

    if(df):
        p_obj = pd.DataFrame(p_obj)
    if(gen__pckl):
        to_pkl(p_obj,gen__pckl)
    return p_obj



#stores the particle data provided in a pkl file f, retrievable
def to_pkl(pobj,f):
    fll = open(f,"wb")
    pkl.dump(pobj, fll)
    fll.close()



# retieves particle data from pkl file
def load_particles_from_file(file):
    f = open(file,"rb")
    parts = pkl.load(f)
    return parts
