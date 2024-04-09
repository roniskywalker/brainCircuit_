import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simCircuit(I):
  firings = np.array([[],[]])

  v = -65*np.ones(Ne+Ni)
  u = b*v

  for t in range(len(I)):

    stim = np.hstack((4*np.random.randn(Ne),1*np.random.randn(Ni))) + I[t]

    fired   = np.where(v>=30)[0]
    tmp     = np.stack((np.tile(t,len(fired)),fired))
    firings = np.concatenate((firings,tmp),axis=1)

    v[fired] = c[fired]
    u[fired] = u[fired] + d[fired]

    stim += np.sum(S[:,fired],axis=1)
 
    v += .04*v**2 + 5*v + 140 - u + stim
    u += a*(b*v-u)
  return firings



def plotPopActivity(firings):
  npnts = int(np.max(firings[0,:])+1) # +1 b/c starts at 0

  popact = np.zeros(npnts)
  for ti in range(npnts):
    popact[ti] = np.sum(firings[0,:]==ti) / (Ne+Ni)

  popactX = np.abs(np.fft.fft(popact-np.mean(popact)))
  hz = np.linspace(0,500,int(npnts/2+1))

  fig,ax = plt.subplots(1,3,figsize=(20,5))
  ax[0].plot(firings[0,:],firings[1,:],'k.',markersize=1)
  ax[0].plot(I*50+100,'m',linewidth=2)
  ax[0].set_xlabel('Time (ms)')
  ax[0].set_ylabel('Neuron #')
  ax[0].set_title('Population spiking')
  
  ax[1].plot(popact)
  ax[1].set_xlabel('Time (ms)')
  ax[1].set_ylabel('Proportion of neurons active')
  ax[1].set_title('Time domain')

  ax[2].plot(hz,popactX[:len(hz)],'k',linewidth=3)
  ax[2].set_xlim([0,80])
  ax[2].set_xlabel('Frequency (Hz)')
  ax[2].set_ylabel('Amplitude (a.u.)')
  ax[2].set_title('Frequency domain')
#   plt.show()
  st.pyplot(fig)


def plotPopActivityEI(firings):
  npnts = int(np.max(firings[0,:])+1)

  popact = np.zeros((2,npnts))
  for ti in range(npnts):
    popact[0,ti] = np.sum(firings[0,firings[1,:]<Ne]==ti) / Ne
    popact[1,ti] = np.sum(firings[0,firings[1,:]>=Ne]==ti) / Ni

  popactXE = np.abs(np.fft.fft(popact[0,:]-np.mean(popact[0,:])))
  popactXI = np.abs(np.fft.fft(popact[1,:]-np.mean(popact[1,:])))
  hz = np.linspace(0,500,int(npnts/2+1))

  fig,ax = plt.subplots(1,3,figsize=(15,5))
  ax[0].plot(firings[0,firings[1,:]>=Ne],firings[1,firings[1,:]>=Ne],'r.',markersize=1)
  ax[0].plot(firings[0,firings[1,:]<Ne],firings[1,firings[1,:]<Ne],'g.',markersize=1)
  ax[0].plot(I*50+100,'m',linewidth=2)
  ax[0].set_xlabel('Time (ms)')
  ax[0].set_ylabel('Neuron #')
  ax[0].set_title('Population spiking')
  
  ax[1].plot(popact[1,:],'r',label='I cells')
  ax[1].plot(popact[0,:],'g',label='E cells')
  ax[1].set_xlabel('Time (ms)')
  ax[1].set_ylabel('Proportion of neurons active')
  ax[1].legend()
  ax[1].set_title('Time domain')

  ax[2].plot(hz,popactXI[:len(hz)],'r',linewidth=2)
  ax[2].plot(hz,popactXE[:len(hz)],'g',linewidth=2)
  ax[2].set_xlim([0,80])
  ax[2].set_xlabel('Frequency (Hz)')
  ax[2].set_ylabel('Amplitude (a.u.)')
  ax[2].set_title('Frequency domain')
#   plt.show()
  st.pyplot(fig)


Ne = 800
Ni = 200
re = np.random.rand(Ne)
ri = np.random.rand(Ni)

a = np.hstack((.02*np.ones(Ne), .02+.08*ri))
b = np.hstack((.2*np.ones(Ne), .25-.05*ri))
c = np.hstack((-65+15*re**2, -65*np.ones(Ni)))
d = np.hstack((8-6*re**2, 2*np.ones(Ni)))

v = -65*np.ones(Ne+Ni)
u = b*v

S = np.hstack((.5*np.random.rand(Ne+Ni,Ne), -np.random.rand(Ne+Ni,Ni) ))



options = st.selectbox("Select for different graphs",("option1","option2","option3"))
if options == "option1":
    input1 = int(st.number_input(label="Put option 1"))
    I = np.ones(input1)
    I[400:601] = -2

    # I = np.sin(np.linspace(0,6*np.pi,2000)*2)

    # networkspikes = simCircuit(I)
    # plotPopActivity(networkspikes)

    # I = np.sin(np.linspace(0,6*np.pi,2435)*2)
    # I = np.random.randn(1400)
    # networkspikes = simCircuit(I)
    # plotPopActivityEI(networkspikes)

elif options == "option2":
    input2 = int(st.number_input(label="put option 2 "))
    I = (np.linspace(-2,2,input2))**2

    # I = np.sin(np.linspace(0,6*np.pi,2000)*2)

    # networkspikes = simCircuit(I)
    # plotPopActivity(networkspikes)

    # I = np.sin(np.linspace(0,6*np.pi,2435)*2)
    # I = np.random.randn(1400)
    # networkspikes = simCircuit(I)
    # plotPopActivityEI(networkspikes)

elif options == "option3":
    input3 = int(st.number_input(label="put option 3"))
    try:
        I = np.sin(np.linspace(0,6*np.pi,input3)*2)
        networkspikes = simCircuit(I)
        plotPopActivity(networkspikes)
    except:
       st.write("No input was given")

    # I = np.sin(np.linspace(0,6*np.pi,input3)*2)
    # I = np.random.randn(1400)
    # networkspikes = simCircuit(I)
    # plotPopActivityEI(networkspikes)
    
    