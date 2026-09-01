import numpy as np
import pandas as pd
from pyomo.environ import * 


class BendersEnergyHub:
    def __init__(self, name=None):
        self.Links = []
        self.Buses = []
        self.Lines = []
        self.Profiles = []

        self.LB = -100000
        self.UB = 100000
        self.sb = 30
        self.Time = 24
        self.Big_M = 10000
        self.refB = 1
        self.Inv_MW_PV = 173.6180556
        self.Upper_Bound_PV_Cap = 10
        self.OC_PV = 1.9

    def ReadExcelFile(self, FileName):
        self.Links = pd.read_excel(FileName, sheet_name='Adj_Matrix')
        self.Buses = pd.read_excel(FileName, sheet_name='Buses')
        self.Lines = pd.read_excel(FileName, sheet_name='Lines')
        self.Profiles = pd.read_excel(FileName, sheet_name='Profiles')

        self.Smax = self.Lines.loc[0,'Smax']*self.Links
        self.buses = len(self.Buses)
        self.V_min = np.unique(self.Buses['Vmin'])[0]
        self.V_max = np.unique(self.Buses['Vmax'])[0]

        self.x_matrix = pd.DataFrame(np.zeros([self.buses,self.buses]))

        for i in range(len(self.Lines)):
            self.fbus = self.Lines.loc[i,'fbus']
            self.tbus = self.Lines.loc[i,'tbus']
            self.aux = self.Lines.loc[i,'x']
            self.x_matrix.loc[self.fbus-1,self.tbus-1] = self.aux
            self.x_matrix.loc[self.tbus-1,self.fbus-1] = self.aux

    def Deterministic_Model (self):
        model = AbstractModel(name='Original_Model')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P_'+str(i)]
        model.PL = Param(model.B,model.T, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def PGmax_SUB_init(model,i):
            return self.Buses.loc[i-1,'PG_max_SUB']
        model.PGmax_Sub = Param(model.B, rule=PGmax_SUB_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.X = Param(model.B,model.B,rule=x_init)

        def Price_init(mode,t):
            return self.Profiles.loc[t-1,'Price']
        model.Lambda = Param(model.T,  rule=Price_init)

        def DG_max (model,t):
            return self.Profiles.loc[t-1,'PV_max']
        model.DGmax = Param(model.T, rule=DG_max)

        model.Vmin = Param(model.B, initialize = self.V_min)
        model.Vmax = Param(model.B, initialize = self.V_max)
        model.Big_M = self.Big_M
        model.Inv_PV = self.Inv_MW_PV
        model.U_P = self.Upper_Bound_PV_Cap
        model.sb = self.sb

        model.pg = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T, initialize = 0)
        model.th = Var(model.B,model.T, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,)

        model.x = Var(model.B, within = Binary)
        model.Gamma_P = Var(model.B, within = NonNegativeReals)
        model.pg_pv = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)


        def Fun_obj(model):
            return sum(model.pg[i,t]*model.sb*model.Lambda[t] for i in model.B for t in model.T) +\
                sum((model.Gamma_P[i]*model.sb)*model.Inv_PV + 10*model.x[i] for i in model.B)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,t):
            Value = sum(model.p[k,l,t] for (k,l) in model.L if k==i)
            return Value == model.pg_pv[i,t] - model.PL[i,t]/self.sb + model.pg[i,t]
        model.npfe = Constraint(model.B,model.T, rule=NPFE)

        def GL_PG_1 (model, i,t):
            return model.pg[i,t] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_1 = Constraint(model.B, model.T, rule=GL_PG_1)

        def LPFE (model,i,j,t):
            return model.p[i,j,t] == (model.th[i,t]-model.th[j,t])/model.X[i,j]
        model.lpfe = Constraint(model.L,model.T, rule=LPFE)

        def V_Limit (model,i,t):
            return inequality(model.Vmin[i], model.v[i,t], model.Vmax[i])
        model.v_limit = Constraint(model.B,model.T, rule=V_Limit)

        def Smax_Limit (model,i,j,t):
            return  model.p[i,j,t] <= model.s_max[i,j]/self.sb
        model.smax_limit = Constraint(model.L,model.T, rule=Smax_Limit)

        def SUB_B (model, sub,t):
            return model.th[sub,t] == 0.0
        model.sub_bus = Constraint(model.refB,model.T, rule=SUB_B)

        def GL_PG2 (model, i,t):
            return model.pg_pv[i,t] <= (model.DGmax[t]*model.Gamma_P[i])/model.sb
        model.gl_pg2 = Constraint(model.B, model.T,rule=GL_PG2)

        def PG_CAP (model, i):
            return model.Gamma_P[i] <= (model.U_P)*model.x[i]/model.sb
        model.pg_cap = Constraint(model.B, rule=PG_CAP)

        def PG_CAP2 (model, i):
            return model.Gamma_P[i] >= 1.5*model.x[i]/model.sb
        model.pg_cap2 = Constraint(model.B, rule=PG_CAP2)

        return model.create_instance()
    
    def Solver(self):

        dm = self.Deterministic_Model()
        self.opt = SolverFactory("gurobi")
        results = self.opt.solve(dm)
        results.write()

        print('Objective Function Value: ' + str(value(dm.FunObj)))

        for i in dm.B:
            print('Capacidad Instalada en '+str(i)+' es: ' +str(round(dm.Gamma_P[i].value,2)))
        

if __name__ == "__main__":
    runing = BendersEnergyHub()
    runing.ReadExcelFile('../data/planning_15bus_deterministic.xlsx')
    runing.Solver()
