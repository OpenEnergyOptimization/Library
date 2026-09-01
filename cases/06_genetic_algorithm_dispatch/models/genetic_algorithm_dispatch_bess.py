import numpy as np
import pandas as pd
from pyomo.environ import * 
import random as rm


class GA_EconomicDispatch_BESS:
    def __init__(self, name=None):
        self.Links = []
        self.Buses = []
        self.Lines = []
        self.Profiles = []

        self.LB = -100000
        self.UB = 100000
        self.sb = 100
        self.Time = 24
        self.Big_M = 100
        self.Agents = [2,3,4]
        self.refB = 1
        self.PB = 1.5
        self.SOC_Min = 0.1
        self.SOC_Max = 0.9
        self.Fi = 0.85

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
        model.A = Set(initialize=self.Agents)
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))
        model.TT = Set(initialize=np.arange(self.Time))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P'+str(i)]
        model.PL = Param(model.B,model.T, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def Beta_init(model,i):
            return self.Buses.loc[i-1,'Beta']
        model.Beta = Param(model.B, rule=Beta_init)

        def I_init(model,i):
            return self.Buses.loc[i-1,'I']
        model.I = Param(model.B, rule=I_init)

        def PGmin_init(model,i,t):
            return self.Buses.loc[i-1,'I']*self.Profiles.loc[t-1,'PG_min']
        model.PG_min = Param(model.B,model.T, rule=PGmin_init)

        def PGmax_init(model,i,t):
            return self.Buses.loc[i-1,'I']*self.Profiles.loc[t-1,'PG_max']
        model.PG_max = Param(model.B,model.T, rule=PGmax_init)

        def PGmax_SUB_init(model,i):
            return self.Buses.loc[i-1,'PG_max_SUB']
        model.PGmax_Sub = Param(model.B, rule=PGmax_SUB_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.x = Param(model.B,model.B,rule=x_init)

        def Price_init(mode,t):
            return self.Profiles.loc[t-1,'Price']
        model.Lambda = Param(model.T,rule=Price_init)

        def BT_init(model,i):
            return self.Buses.loc[i-1,'BT']
        model.BTcap = Param(model.A, rule=BT_init)

        def Loc_BT_init(model,i):
            return self.Buses.loc[i-1,'Loc_BT']
        model.Loc_BT = Param(model.A, rule=Loc_BT_init)

        model.Vmin = Param(model.B, initialize = self.V_min)
        model.Vmax = Param(model.B, initialize = self.V_max)
        model.Big_M = self.Big_M
        model.PB = self.PB
        model.SOC_min = self.SOC_Min
        model.SOC_max = self.SOC_Max
        model.Fi = self.Fi

        model.kappa_bg = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)
        model.kappa_sg = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)
        model.pg = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T, initialize = 0)
        model.th = Var(model.B,model.T, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,)
        model.ds = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.ch = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.soc = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.w = Var(model.A,model.T, within = Binary, initialize = 0)

        def Fun_obj(model):
            return sum(model.Lambda[t]*(model.kappa_bg[i,t]*self.sb - model.kappa_sg[i,t]*self.sb/3) for i in model.B for t in model.T) 
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,t):
            Value = sum(model.p[k,l,t] for (k,l) in model.L if k==i)
            if i in model.A:
                return Value == model.pg[i,t] - model.PL[i,t]/self.sb + model.ds[i,t] - model.ch[i,t]
            else:
                return Value == model.kappa_bg[i,t]  - model.kappa_sg[i,t]
        model.npfe = Constraint(model.B,model.T, rule=NPFE)

        def GL_PG_1 (model, i,t):
            return model.kappa_bg[i,t] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_1 = Constraint(model.B, model.T, rule=GL_PG_1)

        def GL_PG_2 (model, i,t):
            return model.kappa_sg[i,t] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_2 = Constraint(model.B, model.T, rule=GL_PG_2)

        def LPFE (model,i,j,t):
            return model.p[i,j,t] == (model.th[i,t]-model.th[j,t])/model.x[i,j]
        model.lpfe = Constraint(model.L,model.T, rule=LPFE)

        def Cap_Limit (model,i,t):
            return inequality((model.Beta[i]/self.sb)*model.PG_min[i,t], model.pg[i,t], (model.Beta[i]/self.sb)*model.PG_max[i,t])
        model.cap_limit = Constraint(model.A,model.T, rule=Cap_Limit)

        def V_Limit (model,i,t):
            return inequality(model.Vmin[i], model.v[i,t], model.Vmax[i])
        model.v_limit = Constraint(model.B,model.T, rule=V_Limit)

        def Smax_Limit (model,i,j,t):
            return  model.p[i,j,t] <= model.s_max[i,j]/self.sb
        model.smax_limit = Constraint(model.L,model.T, rule=Smax_Limit)

        def SUB_B (model, sub,t):
            return model.th[sub,t] == 0.0
        model.sub_bus = Constraint(model.refB,model.T, rule=SUB_B)

        def SOCR (model,i,t):
            if t==0:
                return model.soc[i,t+1] == 0.5*(model.BTcap[i]/self.sb)
            else:
                return model.soc[i,t+1] == model.soc[i,t] + model.Fi*(model.ch[i,t]) - (model.ds[i,t])/(model.Fi)
        model.socr = Constraint(model.A,model.TT, rule=SOCR)

        def Limit_SOC (model,i,t): 
            return inequality((model.BTcap[i]/self.sb)*model.SOC_min, model.soc[i,t],(model.BTcap[i]/self.sb)*model.SOC_max)
        model.limit_soc = Constraint(model.A,model.T, rule=Limit_SOC)

        def SOCR2 (model,i,t): 
            return (model.ds[i,t]) + model.SOC_min*(model.BTcap[i]/self.sb) <= model.soc[i,t]
        model.socr2 = Constraint(model.A,model.T, rule=SOCR2)

        def CHR (model,i,t): 
            return (model.ch[i,t]) <= (model.PB/self.sb)*model.w[i,t]
        model.chr = Constraint(model.A,model.T, rule=CHR)

        def DSR (model,i,t): 
            return (model.ds[i,t]) <= (model.PB/self.sb)*(1-model.w[i,t])
        model.dsr = Constraint(model.A,model.T, rule=DSR)

        def WR (model,i,t): 
            return model.w[i,t] <= (model.BTcap[i])*1000
        model.wr = Constraint(model.A,model.T, rule=WR)

        return model.create_instance()
    
    def ProblemForGA(self):
        model = AbstractModel(name='Model_for_GA')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.A = Set(initialize=self.Agents)
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))
        model.TT = Set(initialize=np.arange(self.Time))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P'+str(i)]
        model.PL = Param(model.B,model.T, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def Beta_init(model,i):
            return self.Buses.loc[i-1,'Beta']
        model.Beta = Param(model.B, rule=Beta_init)

        def I_init(model,i):
            return self.Buses.loc[i-1,'I']
        model.I = Param(model.B, rule=I_init)

        def PGmin_init(model,i,t):
            return self.Buses.loc[i-1,'I']*self.Profiles.loc[t-1,'PG_min']
        model.PG_min = Param(model.B,model.T, rule=PGmin_init)

        def PGmax_init(model,i,t):
            return self.Buses.loc[i-1,'I']*self.Profiles.loc[t-1,'PG_max']
        model.PG_max = Param(model.B,model.T, rule=PGmax_init)

        def PGmax_SUB_init(model,i):
            return self.Buses.loc[i-1,'PG_max_SUB']
        model.PGmax_Sub = Param(model.B, rule=PGmax_SUB_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.x = Param(model.B,model.B,rule=x_init)

        def Price_init(mode,t):
            return self.Profiles.loc[t-1,'Price']
        model.Lambda = Param(model.T,rule=Price_init)

        def BT_init(model,i):
            return self.Buses.loc[i-1,'BT']
        model.BTcap = Param(model.A, rule=BT_init)

        def Loc_BT_init(model,i):
            return self.Buses.loc[i-1,'Loc_BT']
        model.Loc_BT = Param(model.A, rule=Loc_BT_init)

        model.Vmin = Param(model.B, initialize = self.V_min)
        model.Vmax = Param(model.B, initialize = self.V_max)
        model.Big_M = self.Big_M
        model.PB = self.PB
        model.SOC_min = self.SOC_Min
        model.SOC_max = self.SOC_Max
        model.Fi = self.Fi

        model.kappa_bg = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)
        model.kappa_sg = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)
        model.pg = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T, initialize = 0)
        model.th = Var(model.B,model.T, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,)
        model.ds = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.ch = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.soc = Var(model.A,model.T, within = NonNegativeReals, initialize = 0)
        model.w = Var(model.A,model.T, within = Binary, initialize = 0)

        def Fun_obj(model):
            return sum(model.Lambda[t]*(model.kappa_bg[i,t]*self.sb - model.kappa_sg[i,t]*self.sb/3) for i in model.B for t in model.T) 
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,t):
            Value = sum(model.p[k,l,t] for (k,l) in model.L if k==i)
            if i in model.A:
                return Value == model.pg[i,t] - model.PL[i,t]/self.sb + model.ds[i,t] - model.ch[i,t]
            else:
                return Value == model.kappa_bg[i,t]  - model.kappa_sg[i,t]
        model.npfe = Constraint(model.B,model.T, rule=NPFE)

        def GL_PG_1 (model, i,t):
            return model.kappa_bg[i,t] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_1 = Constraint(model.B, model.T, rule=GL_PG_1)

        def GL_PG_2 (model, i,t):
            return model.kappa_sg[i,t] <= model.PGmax_Sub[i]/self.sb
        model.gl_pg_2 = Constraint(model.B, model.T, rule=GL_PG_2)

        def LPFE (model,i,j,t):
            return model.p[i,j,t] == (model.th[i,t]-model.th[j,t])/model.x[i,j]
        model.lpfe = Constraint(model.L,model.T, rule=LPFE)

        def Cap_Limit (model,i,t):
            return inequality((model.Beta[i]/self.sb)*model.PG_min[i,t], model.pg[i,t], (model.Beta[i]/self.sb)*model.PG_max[i,t])
        model.cap_limit = Constraint(model.A,model.T, rule=Cap_Limit)

        def V_Limit (model,i,t):
            return inequality(model.Vmin[i], model.v[i,t], model.Vmax[i])
        model.v_limit = Constraint(model.B,model.T, rule=V_Limit)

        def Smax_Limit (model,i,j,t):
            return  model.p[i,j,t] <= model.s_max[i,j]/self.sb
        model.smax_limit = Constraint(model.L,model.T, rule=Smax_Limit)

        def SUB_B (model, sub,t):
            return model.th[sub,t] == 0.0
        model.sub_bus = Constraint(model.refB,model.T, rule=SUB_B)

        def SOCR (model,i,t):
            if t==0:
                return model.soc[i,t+1] == 0.5*(model.BTcap[i]/self.sb)
            else:
                return model.soc[i,t+1] == model.soc[i,t] + model.Fi*(model.ch[i,t]) - (model.ds[i,t])/(model.Fi)
        model.socr = Constraint(model.A,model.TT, rule=SOCR)

        def Limit_SOC (model,i,t): 
            return inequality((model.BTcap[i]/self.sb)*model.SOC_min, model.soc[i,t],(model.BTcap[i]/self.sb)*model.SOC_max)
        model.limit_soc = Constraint(model.A,model.T, rule=Limit_SOC)

        def SOCR2 (model,i,t): 
            return (model.ds[i,t]) + model.SOC_min*(model.BTcap[i]/self.sb) <= model.soc[i,t]
        model.socr2 = Constraint(model.A,model.T, rule=SOCR2)

        def CHR (model,i,t): 
            return (model.ch[i,t]) <= (model.PB/self.sb)*model.w[i,t]
        model.chr = Constraint(model.A,model.T, rule=CHR)

        def DSR (model,i,t): 
            return (model.ds[i,t]) <= (model.PB/self.sb)*(1-model.w[i,t])
        model.dsr = Constraint(model.A,model.T, rule=DSR)

        def WR (model,i,t): 
            return model.w[i,t] <= (model.BTcap[i])*1000
        model.wr = Constraint(model.A,model.T, rule=WR)

        return model.create_instance()
    
    def RunOriginalModel(self):

        op = self.OriginalModel()
        self.opt = SolverFactory("gurobi")
        self.opt.solve(op)
        Results = value(op.FunObj)
        print('Objective Function Original Model: ' +str(Results))
        return op


    def GA(self):

        Iteration = 50
        max_no_improvement = 15
        no_improvement = 0  
        k=0
        Population = 6
        Best_FF = 7
        Dict_W, Dict_FF, Dict_FF_Sorted, Dict_W_Best = {},{},{},{}

        ga = self.ProblemForGA()

        self.opt = SolverFactory("gurobi")

        for p in range(1,Population+1):
            for i in ga.A:
                for t in ga.T:
                    if ga.BTcap[i]>0:
                        Dict_W[p,i,t] = np.random.choice([0,1],p=[0.1, 0.9])

                        ga.w[i,t].fixed = True
                        ga.w[i,t].value = Dict_W[p,i,t]
                    else:
                        Dict_W[p,i,t] = 0
                        ga.w[i,t].fixed = True
                        ga.w[i,t].value = Dict_W[p,i,t]

            self.opt.solve(ga)

            Dict_FF[p] = value(ga.FunObj)
            if Dict_FF[p] < Best_FF:
                Best_FF = Dict_FF[p]

        print('Best Initial Value: ' +str(Best_FF))

        while k<Iteration and no_improvement < max_no_improvement:
            k+=1 
            Dict_FF_Sorted = dict(sorted(Dict_FF.items(), key=lambda item: item[1]))
            Population_Sorted = list(Dict_FF_Sorted.keys())
            Pop1 = Population_Sorted[::2]
            Pop2 = Population_Sorted[1::2]
            Pair_Pop = list(zip(Pop1,Pop2))

            for (p1,p2) in Pair_Pop:
                for i in ga.A:
                    for t in ga.T:
                        if t <= 12:
                            Dict_W[p1,i,t] = Dict_W[p2,i,t]
                            Dict_W[p2,i,t] = Dict_W[p1,i,t]

            for p in range(1,Population+1):
                for i in ga.A:
                    if ga.BTcap[i]>0:
                        Random_Sample = rm.sample(range(1,25),4)
                        for tt in Random_Sample:
                            if Dict_W[p,i,tt] == 0:
                                Dict_W[p,i,tt] = 1
                            else:
                                Dict_W[p,i,tt] = 0

            improvement = False

            for p in range(1,Population+1):
                for i in ga.A:
                    for t in ga.T:
                        if ga.BTcap[i]>0:

                            ga.w[i,t].fixed = True
                            ga.w[i,t].value = Dict_W[p,i,t]
                        else:
                            Dict_W[p,i,t] = 0
                            ga.w[i,t].fixed = True
                            ga.w[i,t].value = Dict_W[p,i,t]

                self.opt.solve(ga)

                Dict_FF[p] = value(ga.FunObj)
                if Dict_FF[p] < Best_FF:
                    Best_FF = Dict_FF[p]
                    improvement = True 
                    for i in ga.A:
                        for t in ga.T:
                            Dict_W_Best[i,t] = Dict_W[p,i,t]

                else:
                    for i in ga.A:
                        for t in ga.T:
                            if ga.BTcap[i]>0 and value(ga.w[i,t])==0 and value(ga.ds[i,t]) == 0:
                                Dict_W[p,i,t] = 1

                            if ga.BTcap[i]>0 and value(ga.w[i,t])==1 and value(ga.ch[i,t]) == 0:
                                Dict_W[p,i,t] = 0

            if improvement:
                no_improvement = 0
            else:
                no_improvement += 1

            print('Best Value: ' +str(Best_FF) + ' Iteration: ' +str(k))

        for i in ga.A:
            for t in ga.T:
                if ga.BTcap[i]>0:

                    ga.w[i,t].fixed = True
                    ga.w[i,t].value = Dict_W_Best[i,t]
                else:
                    Dict_W[p,i,t] = 0
                    ga.w[i,t].fixed = True
                    ga.w[i,t].value = Dict_W_Best[i,t]

        self.opt.solve(ga)
        print(value(ga.FunObj))
        return ga

                
    def PrintResults(self,rp):
        PGrid_Table_BG = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])
        PGrid_Table_SG = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])
        PV_Table = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])
        DS_Table = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])
        CH_Table = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])
        W_Table = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])



        r=0
        for i in rp.A:
            if rp.Beta[i] > 0:
                for t in rp.T:
                    PV_Table.loc[r,'Bus'] = i
                    PV_Table.loc[r,t] = round(rp.pg[i,t].value,4)*self.sb
            r += 1

        r=0
        for i in rp.A:
            if rp.BTcap[i] > 0:
                for t in rp.T:
                    DS_Table.loc[r,'Bus'] = i
                    DS_Table.loc[r,t] = round(rp.ds[i,t].value,4)*self.sb

                    CH_Table.loc[r,'Bus'] = i
                    CH_Table.loc[r,t] = round(rp.ch[i,t].value,4)*self.sb

                    W_Table.loc[r,'Bus'] = i
                    W_Table.loc[r,t] = rp.w[i,t].value
                r += 1

        with pd.ExcelWriter('../results/results_genetic_algorithm_dispatch_bess.xlsx') as data:
            PGrid_Table_BG.to_excel(data, sheet_name="Kappa_BG")
            PGrid_Table_SG.to_excel(data, sheet_name="Kappa_SG")
            PV_Table.to_excel(data, sheet_name="PV")
            DS_Table.to_excel(data, sheet_name="DS")
            CH_Table.to_excel(data, sheet_name="CH")
            W_Table.to_excel(data, sheet_name="W")


if __name__ == "__main__":
    runing = GA_EconomicDispatch_BESS()
    runing.ReadExcelFile('../data/ga_5bus_system.xlsx')
    runing.RunOriginalModel()
    results = runing.GA()
    runing.PrintResults(results)
