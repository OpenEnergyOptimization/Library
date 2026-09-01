import numpy as np
import pandas as pd
from pyomo.environ import * 
import pyomo.environ as py

class SOCP_ACOPF_Model:
    def __init__(self, name=None):
        self.data = {}

    def ReadExcelFile(self, FileName):
        self.Data_Ybus = pd.read_excel(FileName, sheet_name='Input_Ybus')
        self.Lines = pd.read_excel(FileName, sheet_name='Adj_Matrix')
        self.Bus_Data = pd.read_excel(FileName, sheet_name='Buses')
        self.Costs_Data = pd.read_excel(FileName, sheet_name='Costs')
        self.Branch_Data = pd.read_excel(FileName, sheet_name='Lines')

        self.NBus = len(self.Bus_Data)
        self.refB = int(self.Bus_Data['refB'][0])

        self.X_Matrix = pd.DataFrame(np.zeros([self.NBus,self.NBus]))
        self.R_Matrix = pd.DataFrame(np.zeros([self.NBus,self.NBus]))
        self.Smax = pd.DataFrame(np.zeros([self.NBus,self.NBus]))

        for i in range(len(self.Branch_Data)):
            self.fbus = self.Branch_Data.loc[i,'fbus']
            self.tbus = self.Branch_Data.loc[i,'tbus']
            self.X_Matrix.loc[self.fbus-1,self.tbus-1] = self.Branch_Data.loc[i,'x']
            self.R_Matrix.loc[self.fbus-1,self.tbus-1] = self.Branch_Data.loc[i,'r']
            self.Smax.loc[self.fbus-1,self.tbus-1] = self.Branch_Data.loc[i,'Smax']

        self.sb = 100

    def Model(self):
        model = AbstractModel()

        model.B = py.Set(initialize=np.arange(1,self.NBus+1))
        model.refB = py.Set(initialize=np.arange(self.refB,self.refB+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.NBus) for j in np.arange(0,self.NBus) if self.Lines.loc[i,j]==1 if i<j)
        model.L = py.Set(dimen=2,initialize=Set_L_init)
            
        model.PGmin = py.Param(model.B, initialize = 0)

        def Ca_init(model,i):
            return self.Costs_Data['a'].loc[i-1]
        model.Ca = py.Param(model.B, rule=Ca_init)

        def Cb_init(model,i):
            return self.Costs_Data['b'].loc[i-1]
        model.Cb = py.Param(model.B, rule=Cb_init)

        def Cc_init(model,i):
            return self.Costs_Data['c'].loc[i-1]
        model.Cc = py.Param(model.B, rule=Cc_init)

        def PL_init(model,i):
            return self.Bus_Data['PL'].loc[i-1]
        model.PL = py.Param(model.B, rule=PL_init)

        def QL_init(model,i):
            return self.Bus_Data['QL'].loc[i-1]
        model.QL = py.Param(model.B, rule=QL_init)

        def PGmax_init(model,i):
            return self.Bus_Data['PGmax'].loc[i-1]
        model.PGmax = py.Param(model.B, rule=PGmax_init)

        def QGmin_init(model,i):
            return self.Bus_Data['QGmin'].loc[i-1]
        model.QGmin = py.Param(model.B,rule=QGmin_init)

        def QGmax_init(model,i):
            return self.Bus_Data['QGmax'].loc[i-1]
        model.QGmax = py.Param(model.B, rule=QGmax_init)

        def Vmax_init(model,i):
            return self.Bus_Data['Vmax'].loc[i-1]
        model.Vmax = py.Param(model.B, rule=Vmax_init)

        def Vmin_init(model,i):
            return self.Bus_Data['Vmin'].loc[i-1]
        model.Vmin = py.Param(model.B, rule=Vmin_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = py.Param(model.B,model.B,rule=Smax_init)

        def X_init(model,i,j):
            return self.X_Matrix.loc[i-1,j-1]
        model.X = py.Param(model.B,model.B,rule=X_init)

        def R_init(model,i,j):
            return self.R_Matrix.loc[i-1,j-1]
        model.R = py.Param(model.B,model.B,rule=R_init)

        model.sb = py.Param(initialize = self.sb)

        model.pg = py.Var(model.B, within = py.NonNegativeReals, initialize = 0)
        model.qg = py.Var(model.B,initialize = 0)
        model.V = py.Var(model.B, initialize = 0, within=py.NonNegativeReals)
        model.P = py.Var(model.L, initialize = 0)
        model.Q = py.Var(model.L, initialize = 0)
        model.l = py.Var(model.L, within=py.NonNegativeReals)

        def Fun_obj(model):
            return sum(model.Ca[i]*(model.pg[i]*self.sb)**2 + model.Cb[i]*(model.pg[i]*self.sb) + model.Cc[i] for i in model.B)
        model.FunObj = py.Objective(rule = Fun_obj, sense = py.minimize)

        def NPFE_R (model,i):
            value1 = sum(model.P[m,n] for (m,n) in model.L if m==i) - sum(model.P[n,m] - model.R[n,m]*model.l[n,m] for (n,m) in model.L if m==i) 
            return model.pg[i] - model.PL[i]/self.sb == value1
        model.npfe_r = py.Constraint(model.B, rule=NPFE_R)

        def NPFE_Im (model,i):
            value2 = sum(model.Q[m,n] for (m,n) in model.L if m==i) - sum(model.Q[n,m] - model.X[n,m]*model.l[n,m] for (n,m) in model.L if m==i) 
            return model.qg[i] - model.QL[i]/self.sb == value2
        model.npfe_im = py.Constraint(model.B, rule=NPFE_Im)

        def LPFE_R (model, i,j):
            return model.V[j] == model.V[i] - 2*(model.R[i,j]*model.P[i,j] + model.X[i,j]*model.Q[i,j]) + (model.R[i,j]**2 + model.X[i,j]**2)*model.l[i,j]
        model.lpfe_r = py.Constraint(model.L, rule=LPFE_R)

        def LPFE_Iq (model, i,j):
            return (model.P[i,j]**2 + model.Q[i,j]**2) <= model.l[i,j]*model.V[i]
        model.lpfe_iq = py.Constraint(model.L, rule=LPFE_Iq)

        def GL_PG (model, i):
            return model.pg[i] <= model.PGmax[i]/self.sb
        model.gl_pg = py.Constraint(model.B, rule=GL_PG)

        def GL_QG (model, i):
            return inequality(model.QGmin[i]/self.sb, model.qg[i],model.QGmax[i]/self.sb)
        model.gl_qg = py.Constraint(model.B, rule=GL_QG)

        def LSL (model,i,j):
            return (model.P[i,j]**2 + model.Q[i,j]**2) <= (model.s_max[i,j]/self.sb)**2
        model.lsl = py.Constraint(model.L, rule=LSL)

        def VL (model,i):
            return inequality(model.Vmin[i]**2, model.V[i], model.Vmax[i]**2)
        model.vl = py.Constraint(model.B, rule=VL)

        def SUB_B (model, sub):
            return model.V[sub] == 1
        model.sub_bus = py.Constraint(model.refB, rule=SUB_B)

        return model.create_instance()

    def Solver(self):
        Model = self.Model()
        self.opt = SolverFactory('gurobi')
        results_Model = self.opt.solve(Model)
        results_Model.write()
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        return Model

    def ExportResults(self, model):

        Flow_lines_table = pd.DataFrame(columns=['from','to','Pij', 'Qij','Flow_S','Smax'])
        Power_table = pd.DataFrame(columns=['bus','PG', 'QG','PL','QL'])
        Data_table = pd.DataFrame(columns=['bus','Voltage']) 

        r=0
        for i in model.B:
            Power_table.loc[r,'bus'] = str(i)
            Power_table.loc[r,'PG'] = round(model.pg[i].value,4)*self.sb
            Power_table.loc[r,'QG'] = round(model.qg[i].value,4)*self.sb
            Power_table.loc[r,'PL'] = round(model.PL[i],4)
            Power_table.loc[r,'QL'] = round(model.QL[i],4)
            r += 1

        r=0
        for i in model.B:
            Data_table.loc[r,'bus'] = str(i)
            Data_table.loc[r,'Voltage'] = round(py.sqrt(model.V[i].value),4)
            r += 1

        r=0
        for (i,j) in model.L:
            Flow_lines_table.loc[r,'from'] = str(i)
            Flow_lines_table.loc[r,'to'] = str(j)
            Flow_lines_table.loc[r,'Pij'] = round(model.P[i,j].value,4)*self.sb
            Flow_lines_table.loc[r,'Qij'] = round(model.Q[i,j].value,4)*self.sb
            Flow_lines_table.loc[r,'Flow_S'] = round(sqrt((model.P[i,j].value)**2+(model.Q[i,j].value)**2)*self.sb,4)
            Flow_lines_table.loc[r,'Smax'] = model.s_max[i,j]
            r += 1  

        with pd.ExcelWriter('../results/results_ac_opf_socp.xlsx') as data: 
            Power_table.to_excel(data, sheet_name="Power_injected")
            Data_table.to_excel(data, sheet_name="Volt_Angle")
            Flow_lines_table.to_excel(data, sheet_name="Flow_Lines")

if __name__ == '__main__':
    running = SOCP_ACOPF_Model()
    running.ReadExcelFile('../data/opf_33bus_distribution_system.xlsx')
    modelo = running.Solver()
    excel =  running.ExportResults(modelo)