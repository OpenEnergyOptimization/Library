import numpy as np
import pandas as pd
from pyomo.environ import * 


class BendersEnergyHub:
    def __init__(self, name=None):
        self.Links = []
        self.Buses = []
        self.Lines = []
        self.Profiles = []

        self.LB = -1000000
        self.UB = 1000000
        self.sb = 30
        self.Time = 24
        self.Big_M = 10000
        self.refB = 1
        self.Inv_MW_PV = 173.6180556
        self.Upper_Bound_PV_Cap = 10
        self.Scenarios = 3
        self.OC_PV = 1.9
        self.Rho = [1 / self.Scenarios] * self.Scenarios

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

    def OriginalModel (self):
        model = AbstractModel(name='Original_Model')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))
        model.S = Set(initialize=np.arange(1,self.Scenarios+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t,s):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P_'+str(s)+'_'+str(i)]
        model.PL = Param(model.B,model.T,model.S, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def PGmax_SUB_init(model,i):
            return self.Buses.loc[i-1,'PG_max_SUB']
        model.PGmax_Sub = Param(model.B, rule=PGmax_SUB_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.X = Param(model.B,model.B,rule=x_init)

        def Price_init(mode,t,s):
            return self.Profiles.loc[t-1,'Price_'+str(1)]
        model.Lambda = Param(model.T, model.S, rule=Price_init)

        def DG_max (model,t,s):
            return self.Profiles.loc[t-1,'PV_max_'+str(s)]
        model.DGmax = Param(model.T, model.S, rule=DG_max)

        model.Vmin = Param(model.B, initialize = self.V_min)
        model.Vmax = Param(model.B, initialize = self.V_max)
        model.Big_M = self.Big_M
        model.Inv_PV = self.Inv_MW_PV
        model.U_P = self.Upper_Bound_PV_Cap
        model.sb = self.sb

        model.pg = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T,model.S, initialize = 0)
        model.th = Var(model.B,model.T,model.S, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,model.S,)

        model.x = Var(model.B, within = Binary)
        model.Gamma_P = Var(model.B, within = NonNegativeReals)
        model.pg_pv = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)


        def Fun_obj(model):
            return sum(model.pg[i,t,s]*model.sb*model.Lambda[t,s] for i in model.B for t in model.T for s in model.S) +\
                sum((model.Gamma_P[i]*model.sb)*model.Inv_PV + 10*model.x[i] for i in model.B)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,t,s):
            Value = sum(model.p[k,l,t,s] for (k,l) in model.L if k==i)
            return Value == model.pg_pv[i,t,s] - model.PL[i,t,s]/self.sb + model.pg[i,t,s]
        model.npfe = Constraint(model.B,model.T,model.S, rule=NPFE)

        def GL_PG_1 (model, i,t,s):
            return model.pg[i,t,s] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_1 = Constraint(model.B, model.T,model.S, rule=GL_PG_1)

        def LPFE (model,i,j,t,s):
            return model.p[i,j,t,s] == (model.th[i,t,s]-model.th[j,t,s])/model.X[i,j]
        model.lpfe = Constraint(model.L,model.T,model.S, rule=LPFE)

        def V_Limit (model,i,t,s):
            return inequality(model.Vmin[i], model.v[i,t,s], model.Vmax[i])
        model.v_limit = Constraint(model.B,model.T,model.S, rule=V_Limit)

        def Smax_Limit (model,i,j,t,s):
            return  model.p[i,j,t,s] <= model.s_max[i,j]/self.sb
        model.smax_limit = Constraint(model.L,model.T,model.S, rule=Smax_Limit)

        def SUB_B (model, sub,t,s):
            return model.th[sub,t,s] == 0.0
        model.sub_bus = Constraint(model.refB,model.T,model.S, rule=SUB_B)

        def GL_PG2 (model, i,t,s):
            return model.pg_pv[i,t,s] <= (model.DGmax[t,s]*model.Gamma_P[i])/model.sb
        model.gl_pg2 = Constraint(model.B, model.T,model.S, rule=GL_PG2)

        def PG_CAP (model, i):
            return model.Gamma_P[i] <= (model.U_P)*model.x[i]/model.sb
        model.pg_cap = Constraint(model.B, rule=PG_CAP)

        def PG_CAP2 (model, i):
            return model.Gamma_P[i] >= 1.5*model.x[i]/model.sb
        model.pg_cap2 = Constraint(model.B, rule=PG_CAP2)

        return model.create_instance()
    
    def MasterProblem (self):
        model = AbstractModel(name='Master_Problem')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))
        model.S = Set(initialize=np.arange(1,self.Scenarios+1))

        model.Inv_PV = self.Inv_MW_PV
        model.U_P = self.Upper_Bound_PV_Cap
        model.sb = self.sb

        model.x = Var(model.B, within = Binary)
        model.Gamma_P = Var(model.B, initialize=0, within = NonNegativeReals)
        model.theta = Var(within = Reals, initialize=0, bounds=(-10000000,10000000))

        def Fun_obj(model):
            return sum((model.Gamma_P[i]*model.sb)*model.Inv_PV + 10*model.x[i] for i in model.B) + model.theta
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def PG_CAP (model, i):
            return model.Gamma_P[i] <= (model.U_P)*model.x[i]/model.sb
        model.pg_cap = Constraint(model.B, rule=PG_CAP)

        def PG_CAP2 (model, i):
            return model.Gamma_P[i] >= 1.5*model.x[i]/model.sb
        model.pg_cap2 = Constraint(model.B, rule=PG_CAP2)

        model.benders_cuts = ConstraintList()

        return model.create_instance()

    def SubProblem (self,ss):

        model = AbstractModel(name='Sub_Problem')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))
        model.S = Set(initialize=np.array([ss]))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t,s):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P_'+str(s)+'_'+str(i)]
        model.PL = Param(model.B,model.T,model.S, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def PGmax_SUB_init(model,i):
            return self.Buses.loc[i-1,'PG_max_SUB']
        model.PGmax_Sub = Param(model.B, rule=PGmax_SUB_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.X = Param(model.B,model.B,rule=x_init)

        def Price_init(mode,t,s):
            return self.Profiles.loc[t-1,'Price_'+str(1)]
        model.Lambda = Param(model.T, model.S, rule=Price_init)

        def DG_max (model,t,s):
            return self.Profiles.loc[t-1,'PV_max_'+str(s)]
        model.DGmax = Param(model.T, model.S, rule=DG_max)

        def RHO_Init (model,s):
            return self.Rho[s-1]
        model.Rho = Param(model.S, rule=RHO_Init)

        model.Vmin = Param(model.B, initialize = self.V_min)
        model.Vmax = Param(model.B, initialize = self.V_max)
        model.Big_M = self.Big_M
        model.sb = self.sb
        model.OC_PV = self.OC_PV

        model.pg = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T,model.S, initialize = 0)
        model.th = Var(model.B,model.T,model.S, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,model.S,initialize = 0)

        model.Gamma_P_hat = Var(model.B, within = NonNegativeReals)
        model.pg_pv = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)

        model.Gamma_P_param = Param(model.B, initialize = 0, mutable=True)


        def Fun_obj(model):
            return sum(model.Rho[s]*model.pg[i,t,s]*model.sb*model.Lambda[t,s] +\
                       model.Rho[s]*model.pg_pv[i,t,s]*model.OC_PV*model.sb  for i in model.B for t in model.T for s in model.S)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,t,s):
            Value = sum(model.p[k,l,t,s] for (k,l) in model.L if k==i)
            return Value == model.pg_pv[i,t,s] - model.PL[i,t,s]/self.sb + model.pg[i,t,s]
        model.npfe = Constraint(model.B,model.T,model.S, rule=NPFE)

        def GL_PG_1 (model, i,t,s):
            return model.pg[i,t,s] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_1 = Constraint(model.B, model.T,model.S, rule=GL_PG_1)

        def LPFE (model,i,j,t,s):
            return model.p[i,j,t,s] == (model.th[i,t,s]-model.th[j,t,s])/model.X[i,j]
        model.lpfe = Constraint(model.L,model.T,model.S, rule=LPFE)

        def V_Limit (model,i,t,s):
            return inequality(model.Vmin[i], model.v[i,t,s], model.Vmax[i])
        model.v_limit = Constraint(model.B,model.T,model.S, rule=V_Limit)

        def Smax_Limit (model,i,j,t,s):
            return  model.p[i,j,t,s] <= model.s_max[i,j]/self.sb
        model.smax_limit = Constraint(model.L,model.T,model.S, rule=Smax_Limit)

        def SUB_B (model, sub,t,s):
            return model.th[sub,t,s] == 0.0
        model.sub_bus = Constraint(model.refB,model.T,model.S, rule=SUB_B)

        def GL_PG2 (model, i,t,s):
            return model.pg_pv[i,t,s] <= (model.DGmax[t,s]*model.Gamma_P_hat[i])
        model.gl_pg2 = Constraint(model.B, model.T,model.S, rule=GL_PG2)

        def Gamma_P_Hat (model, i):
            return model.Gamma_P_hat[i] == model.Gamma_P_param[i]
        model.gamma_p_hat_Cons = Constraint(model.B, rule=Gamma_P_Hat)

        return model.create_instance()
    
    def DualProblem (self):
        model = AbstractModel(name='Dual_Problem')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))
        model.S = Set(initialize=np.arange(1,self.Scenarios+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t,s):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P_'+str(s)+'_'+str(i)]
        model.PL = Param(model.B,model.T,model.S, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def PGmax_SUB_init(model,i):
            return self.Buses.loc[i-1,'PG_max_SUB']
        model.PGmax_Sub = Param(model.B, rule=PGmax_SUB_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.X = Param(model.B,model.B,rule=x_init)

        def Price_init(mode,t,s):
            return self.Profiles.loc[t-1,'Price_'+str(1)]
        model.Lambda = Param(model.T, model.S, rule=Price_init)

        def DG_max (model,t,s):
            return self.Profiles.loc[t-1,'PV_max_'+str(s)]
        model.DGmax = Param(model.T, model.S, rule=DG_max)

        def RHO_Init (model,s):
            return self.Rho[s-1]
        model.Rho = Param(model.S, rule=RHO_Init)

        model.Gamma_P_hat = Param(model.B, initialize = 0, mutable=True)

        model.Vmin = Param(model.B, initialize = self.V_min)
        model.Vmax = Param(model.B, initialize = self.V_max)
        model.Big_M = self.Big_M
        model.sb = self.sb
        model.OC_PV = self.OC_PV

        model.pg = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T,model.S, initialize = 0)
        model.th = Var(model.B,model.T,model.S, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,model.S,)
        model.pg_pv = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)
        self.Theta_min = -8*atan(1)
        self.Theta_max = 8*atan(1)

        model.a_dual = Var(model.B,model.T,model.S, within = Reals, initialize = 0)
        model.b_dual = Var(model.L,model.T,model.S, within = Reals, initialize = 0)
        model.c_max_dual = Var(model.L,model.T,model.S, within = NonPositiveReals, initialize = 0)
        model.d_max_dual = Var(model.B,model.T,model.S, within = NonPositiveReals, initialize = 0)
        model.e_min_dual = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)
        model.e_max_dual = Var(model.B,model.T,model.S, within = NonPositiveReals, initialize = 0)
        model.f_max_dual = Var(model.B,model.T,model.S, within = NonPositiveReals, initialize = 0)
        model.g_min_dual = Var(model.B,model.T,model.S, within = NonNegativeReals, initialize = 0)
        model.g_max_dual = Var(model.B,model.T,model.S, within = NonPositiveReals, initialize = 0)


        def Fun_obj(model):
            return sum(-(model.PL[i,t,s]/model.sb)*(model.a_dual[i,t,s]) + \
                    (model.PGmax_Sub[i]/model.sb)*model.d_max_dual[i,t,s] +\
                    model.Vmin[i]*model.e_min_dual[i,t,s] + \
                    model.Vmax[i]*model.e_max_dual[i,t,s] + \
                    model.DGmax[t,s]*(model.Gamma_P_hat[i])*model.f_max_dual[i,t,s] + \
                    self.Theta_min*model.g_min_dual[i,t,s] +\
                    self.Theta_max*model.g_max_dual[i,t,s]          for i in model.B for t in model.T for s in model.S) + \
                    sum( (model.s_max[k,l]/model.sb)*model.c_max_dual[k,l,t,s] for (k,l) in model.L for t in model.T for s in model.S)
        model.FunObj = Objective(rule = Fun_obj, sense = maximize)

        def Dual_Constraints_1 (model,i,t,s):
            return -model.a_dual[i,t,s] + model.d_max_dual[i,t,s] <= model.Lambda[t,s]*model.sb*model.Rho[s]
        model.dual_const_1 = Constraint(model.B,model.T,model.S, rule=Dual_Constraints_1)

        def Dual_Constraints_2 (model,i,t,s):
            return model.e_min_dual[i,t,s] + model.e_max_dual[i,t,s] == 0
        model.dual_const_2 = Constraint(model.B,model.T,model.S, rule=Dual_Constraints_2)

        def Dual_Constraints_3 (model,i,t,s):
            return model.f_max_dual[i,t,s] - model.a_dual[i,t,s] <= model.OC_PV*model.sb*model.Rho[s]
        model.dual_const_3 = Constraint(model.B,model.T,model.S, rule=Dual_Constraints_3)

        def Dual_Constraints_4 (model,i,j,t,s):
            return model.c_max_dual[i,j,t,s] + model.X[i,j]*model.b_dual[i,j,t,s] + model.a_dual[i,t,s] == 0
        model.dual_const_4 = Constraint(model.L,model.T,model.S, rule=Dual_Constraints_4)

        def Dual_Constraints_5 (model,i,t,s):
            return sum(-model.b_dual[k,j,t,s] + model.b_dual[j,k,t,s] for (k,j) in model.L if k==i) + model.g_min_dual[i,t,s] + model.g_max_dual[i,t,s] == 0
        model.dual_const_5 = Constraint(model.B,model.T,model.S, rule=Dual_Constraints_5)


        return model.create_instance()

    def BendersAlgorithm(self):

        mp = self.MasterProblem()
        dsp = self.DualProblem()

        LB, UB    = -np.inf, np.inf 
        tol_gap   = 1e-6
        limititer = 25
        k = 0
        GAP = 10
        self.opt = SolverFactory("gurobi")

        while GAP>tol_gap:
            k += 1

            self.opt.solve(mp)

            LB = value(mp.FunObj)

            for i in mp.B:
                dsp.Gamma_P_hat[i] = mp.Gamma_P[i].value

            self.opt.solve(dsp)

            ni = value(dsp.FunObj)

            UB = min(UB,LB - mp.theta.value + ni)

            mp.benders_cuts.add(mp.theta >= sum(-(dsp.PL[i,t,s]/dsp.sb)*(dsp.a_dual[i,t,s].value) + \
                    (dsp.PGmax_Sub[i]/dsp.sb)*dsp.d_max_dual[i,t,s].value +\
                    dsp.Vmin[i]*dsp.e_min_dual[i,t,s].value + \
                    dsp.Vmax[i]*dsp.e_max_dual[i,t,s].value + \
                    dsp.DGmax[t,s]*(mp.Gamma_P[i])*dsp.f_max_dual[i,t,s].value + \
                    self.Theta_min*dsp.g_min_dual[i,t,s].value +\
                    self.Theta_max*dsp.g_max_dual[i,t,s].value          for i in dsp.B for t in dsp.T for s in dsp.S) + \
                    sum( (dsp.s_max[k,l]/dsp.sb)*dsp.c_max_dual[k,l,t,s].value for (k,l) in dsp.L for t in dsp.T for s in dsp.S))

            GAP = abs(1-LB/UB)

            print("k:%s-----------UB:%s, LB:%s, GAP:%s"%(k,round(UB,6),round(LB,6),round(GAP,6)))

        print('Objective Function Value: ' + str(round(LB,3)))

        PowerFromPV= sum(dsp.pg_pv[i,t,s].value for i in mp.B for t in mp.T for s in mp.S)
        PowerFromSub = sum(dsp.pg[i,t,s].value for i in mp.B for t in mp.T for s in mp.S)
        TotalLoad = sum(dsp.PL[i,t,s] for i in mp.B for t in mp.T for s in mp.S)
        

        for i in mp.B:
            print('Capacidad Instalada en '+str(i)+' es: ' +str(round(mp.Gamma_P[i].value,2)))








        

        

if __name__ == "__main__":
    runing = BendersEnergyHub()
    runing.ReadExcelFile('../data/planning_15bus_stochastic_100scenarios.xlsx')
    runing.BendersAlgorithm()
