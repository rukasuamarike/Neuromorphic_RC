import torch as nn


class MemC:

    #   This class represents a memristor subcircuit with variable capacitance.

    #   Parameters:
    #       Cmin (float): Minimum capacitance value (default: 10nF).
    #       Cmax (float): Maximum capacitance value (default: 10uF).
    #       Cinit (float): Initial capacitance value (default: 100nF).
    #       k (float): Fitting parameter (default: 10meg).
    #       p (float): Fitting parameter (default: 1).
    #       IC (float): Initial voltage across the capacitor (default: 0).

    def __init__(self, Cmin=10e-9, Cmax=10e-6, Cinit=100e-9, k=10e6, p=1, IC=0):

        # Initializes the MemC subcircuit with the specified parameters.
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.Cinit = Cinit
        self.k = k
        self.p = p
        self.IC = IC
        self.xinit = (1 / Cinit - 1 / Cmax) / (1 / Cmin - 1 / Cmax)
        self.xinit = self.Cmin + nn.mul(self.IC, self.Cmax - self.Cmin)
        self.x = xinit  # my MC value
        self.dx= self.IC # my MC0 -> delta MC value
        
        self.value=nn.tensor([self.x,self.dx, self.q,self.dq, self.Cmin, self.Cmax, self.Cinit, self.k, self.p, self.IC])
    def DM(self):
        return 1/self.Cmax + (1/self.Cmin-1/self.Cmax)*self.x
    
    def reset_init(self):
        self.xinit = self.Cinit
        self.xinit = (1 / self.Cinit - 1 / self.Cmax) / (1 / self.Cmin - 1 / self.Cmax)
        self.IC = self.xinit
    
    
    def update_val(self,v,dt):
        self.q=self.x*v # q= c*v
        self.dx=self.x * dt

        self.x=self.q/v
        

        emc = DM(self.x)*(self.q + self.IC*self.Cinit)
