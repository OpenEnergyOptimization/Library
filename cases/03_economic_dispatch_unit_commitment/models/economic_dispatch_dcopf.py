import numpy as np
import pandas as pd
from pyomo.environ import * 


class BendersEnergyHub:
    def __init__(self, name=None):
        self.Links = []
        self.Buses = []
        self.Lines = []
        self.Profiles = []

        self.sb = 100
        self.Time = 24
        self.refB = 1

    def ReadExcelFile(self, FileName):
        self.Links = pd.read_excel(FileName, sheet_name='Adj_Matrix')
        self.Buses = pd.read_excel(FileName, sheet_name='Buses')
        self.Lines = pd.read_excel(FileName, sheet_name='Lines')
        self.Profiles = pd.read_excel(FileName, sheet_name='Profiles')
        self.Costs_Data = pd.read_excel(FileName, sheet_name='Costs')

        self.Smax = self.Lines.loc[0,'Smax']*self.Links
        self.buses = len(self.Buses)
        self.x_matrix = self.Links.copy().astype(float)

        for i in range(len(self.Lines)):
            self.fbus = self.Lines.loc[i,'fbus']
            self.tbus = self.Lines.loc[i,'tbus']
            self.aux = float(self.Lines.loc[i,'x'])
            self.x_matrix.loc[self.fbus-1,self.tbus-1] = self.aux
            self.x_matrix.loc[self.tbus-1,self.fbus-1] = self.aux

    def OriginalModel (self):
        model = AbstractModel(name='Original_Model')

        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))
        model.T = Set(initialize=np.arange(1,self.Time+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1)
        model.L = Set(dimen=2,initialize=Set_L_init)

        def Ca_init(model,i):
            return self.Costs_Data['a'].loc[i-1]
        model.Ca = Param(model.B, rule=Ca_init)

        def Cb_init(model,i):
            return self.Costs_Data['b'].loc[i-1]
        model.Cb = Param(model.B, rule=Cb_init)

        def Cc_init(model,i):
            return self.Costs_Data['c'].loc[i-1]
        model.Cc = Param(model.B, rule=Cc_init)

        def PL_init(model,i,t):
            return self.Buses.loc[i-1,'PL']*self.Profiles.loc[t-1,'P_'+str(i)]
        model.PL = Param(model.B,model.T, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def PGmax_init(model,i):
            return self.Buses.loc[i-1,'PGmax']
        model.PGmax = Param(model.B, rule=PGmax_init)

        def PGmin_init(model,i):
            return self.Buses.loc[i-1,'PGmin']
        model.PGmin = Param(model.B, rule=PGmin_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.X = Param(model.B,model.B,rule=x_init)

        def Vmax_init(model,i):
            return self.Buses.loc[i-1,'Vmax']
        model.Vmax = Param(model.B, rule=Vmax_init)

        def Vmin_init(model,i):
            return self.Buses.loc[i-1,'Vmin']
        model.Vmin = Param(model.B, rule=Vmin_init)


        def Start_up_Cost_init(model,i):
            return self.Costs_Data['Start_Up_Cost'].loc[i-1]
        model.Start_Cost = Param(model.B, rule=Start_up_Cost_init)

        def Ramp_Up_init(model,i):
            return self.Buses['Ramp_Up'].loc[i-1]
        model.Ramp_Up = Param(model.B, rule=Ramp_Up_init)

        def Ramp_Down_init(model,i):
            return self.Buses['Ramp_Down'].loc[i-1]
        model.Ramp_Down = Param(model.B, rule=Ramp_Down_init)


        model.sb = self.sb

        model.pg = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B,model.T, initialize = 0)
        model.th = Var(model.B,model.T, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L,model.T,initialize = 0)

        model.u = Var(model.B,model.T, within = Binary, initialize = 0)
        model.sc = Var(model.B,model.T, within = NonNegativeReals, initialize = 0)


        def Fun_obj(model):
            return sum(model.Ca[i]*(model.pg[i,t]*model.sb)**2 + model.Cb[i]*(model.pg[i,t]*model.sb) + model.Cc[i] for i in model.B for t in model.T)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,t):
            Value = sum(model.p[k,l,t] for (k,l) in model.L if k==i)
            return Value == - model.PL[i,t]/self.sb + model.pg[i,t]
        model.npfe = Constraint(model.B,model.T, rule=NPFE)

        def GL_PG_1 (model, i,t):
            return model.pg[i,t]<= (model.PGmax[i]/self.sb)
        model.gl_pg_1 = Constraint(model.B, model.T, rule=GL_PG_1)

        def GL_PG_2 (model, i,t):
            return model.PGmin[i]/self.sb <= model.pg[i,t]
        model.gl_pg_2 = Constraint(model.B, model.T, rule=GL_PG_2)

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

        return model.create_instance()

    
    def BendersAlgorithm(self):

        op = self.OriginalModel()
        self.opt = SolverFactory("gurobi")
        self.opt.solve(op)
        Results = value(op.FunObj)

        print('Objective Function Value: ' + str(round(Results,3)))

        return op

    def Print_Results(self,op):
        Flow_lines_table = pd.DataFrame()
        PG_Table = pd.DataFrame()
        PL_Table = pd.DataFrame()
        U_Table = pd.DataFrame()

        r=0
        for i in op.B:
            for t in op.T:
                PG_Table.loc[r,'bus'] = str(i)
                PG_Table.loc[r,t] = round(op.pg[i,t].value,4)*self.sb
                PL_Table.loc[r,'bus'] = str(i)
                PL_Table.loc[r,t] = round(op.PL[i,t],4)
                U_Table.loc[r,'bus'] = str(i)
                U_Table.loc[r,t] = round(op.u[i,t].value,4)
            r += 1

        r=0
        for (i,j) in op.L:
            for t in op.T:
                Flow_lines_table.loc[r,'from'] = str(i)
                Flow_lines_table.loc[r,'to'] = str(j)
                Flow_lines_table.loc[r,t] = round(op.p[i,j,t].value,4)*self.sb
                Flow_lines_table.loc[r,'Smax'] = op.s_max[i,j]
            r += 1

        with pd.ExcelWriter('../results/results_economic_dispatch_dcopf.xlsx') as data: 
            PG_Table.to_excel(data, sheet_name="PG")
            PL_Table.to_excel(data, sheet_name="PL")
            U_Table.to_excel(data, sheet_name="U")
            Flow_lines_table.to_excel(data, sheet_name="Flow_Lines")
        

if __name__ == "__main__":
    runing = BendersEnergyHub()
    runing.ReadExcelFile('../data/ed_uc_14bus_system.xlsx')
    op = runing.BendersAlgorithm()
    runing.Print_Results(op)
