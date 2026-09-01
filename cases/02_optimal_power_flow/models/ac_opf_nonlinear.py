import numpy as np
import pandas as pd
from pyomo.environ import * 
from ybus_matrix_builder import Make_New_Ybus
import pyomo.environ as py

class ACOPF_Model:
    def __init__(self, name=None):
        self.data = {}

    def ReadExcelFile(self, FileName):
        self.Data_Ybus = pd.read_excel(FileName, sheet_name='Input_Ybus')
        self.Lines = pd.read_excel(FileName, sheet_name='Adj_Matrix')
        self.Bus_Data = pd.read_excel(FileName, sheet_name='Buses')
        self.Costs_Data = pd.read_excel(FileName, sheet_name='Costs')
        self.sb = 100

    def Model(self):
        model = AbstractModel()

        NBus = len(self.Bus_Data)
        refB = int(self.Bus_Data['refB'][0])
        YBus, Smax, BS = Make_New_Ybus(self.sb,self.Bus_Data,self.Data_Ybus)

        model.B = py.Set(initialize=np.arange(1,NBus+1))
        model.refB = py.Set(initialize=np.arange(refB,refB+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,NBus) for j in np.arange(0,NBus) if self.Lines.loc[i,j]==1)
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

        def BS_init(model,i,j):
            return BS.loc[i-1,j-1]
        model.bs = py.Param(model.B,model.B, rule=BS_init)

        def Smax_init(model,i,j):
            return Smax.loc[i-1,j-1]
        model.s_max = py.Param(model.B,model.B,rule=Smax_init)

        def Gij_init(model,i,j):
            if i != j:
                return round(YBus.loc[i-1,j-1].real,8)
            else:
                return 0
        model.Gij = py.Param(model.B,model.B, within = py.Reals, rule=Gij_init)

        def Bij_init(model,i,j):
            if i != j:
                return round(YBus.loc[i-1,j-1].imag,8)
            else:
                return 0
        model.Bij = py.Param(model.B,model.B, within = py.Reals, rule=Bij_init)

        def Gi_init (model,i):
            return round(YBus.loc[i-1,i-1].real,8)
        model.Gi = py.Param(model.B, within = py.Reals, rule=Gi_init)

        def Bi_init (model,i):
            return round(YBus.loc[i-1,i-1].imag,8)
        model.Bi = py.Param(model.B, within = py.Reals, rule=Bi_init)

        model.sb = py.Param(initialize = self.sb)

        model.pg = py.Var(model.B, within = py.NonNegativeReals, initialize = 0)
        model.qg = py.Var(model.B,initialize = 0)
        model.V = py.Var(model.B, initialize = 0)
        model.Th = py.Var(model.B, bounds = (-8*py.atan(1),8*py.atan(1)))
        model.P = py.Var(model.L, initialize = 1)
        model.Q = py.Var(model.L, initialize = 0)
        model.aux = py.Var(model.B, within = py.Reals, initialize = 0)
        model.aux2 = py.Var(model.B, within = py.Reals, initialize = 0)

        def Fun_obj(model):
            return sum(model.Ca[i]*(model.pg[i]*self.sb)**2 + model.Cb[i]*(model.pg[i]*self.sb) + model.Cc[i] for i in model.B)
        model.FunObj = py.Objective(rule = Fun_obj, sense = py.minimize)

        def NPFE_R (model,i):
            value1 = model.V[i]**2*model.Gi[i] + model.V[i]*sum(model.V[l]*(model.Gij[r,l]*py.cos(model.Th[r]-model.Th[l]) + model.Bij[r,l]*py.sin(model.Th[r]-model.Th[l])) for (r,l) in model.L if r==i)
            return model.pg[i] - model.PL[i]/self.sb == value1
        model.npfe_r = py.Constraint(model.B, rule=NPFE_R)

        def NPFE_Im (model,i):
            value2 = (-1)*model.V[i]**2*model.Bi[i] + model.V[i]*sum(model.V[l]*(model.Gij[r,l]*py.sin(model.Th[r]-model.Th[l]) - model.Bij[r,l]*py.cos(model.Th[r]-model.Th[l])) for (r,l) in model.L if r==i)
            return model.qg[i] - model.QL[i]/self.sb  == value2
        model.npfe_im = py.Constraint(model.B, rule=NPFE_Im)
                
        def LPFE_R (model, i,j):
            value_3 = model.V[i]*model.V[j]*(model.Gij[i,j]*py.cos(model.Th[i]- model.Th[j]) + model.Bij[i,j]*py.sin(model.Th[i]- model.Th[j])) - model.Gij[i,j]*model.V[i]**2
            return model.P[i,j] == value_3
        model.lpfe_r = py.Constraint(model.L, rule=LPFE_R)
            
        def LPFE_Im (model, i,j):
            value_4 = model.V[i]*model.V[j]*(model.Gij[i,j]*py.sin(model.Th[i]- model.Th[j]) - model.Bij[i,j]*py.cos(model.Th[i]- model.Th[j])) + model.V[i]**2*(model.Bij[i,j] - model.bs[i,j])
            return model.Q[i,j] == value_4
        model.lpfe_im = py.Constraint(model.L, rule=LPFE_Im)

        def GL_PG (model, i):
            return model.pg[i] <= model.PGmax[i]/self.sb
        model.gl_pg = py.Constraint(model.B, rule=GL_PG)

        def GL_QG (model, i):
            return py.inequality(model.QGmin[i]/self.sb, model.qg[i],model.QGmax[i]/self.sb)
        model.gl_qg = py.Constraint(model.B, rule=GL_QG)

        def LSL (model,i,j):
            return py.sqrt(model.P[i,j]**2 + model.Q[i,j]**2) <= model.s_max[i,j]/self.sb
        model.lsl = py.Constraint(model.L, rule=LSL)

        def VL (model,i):
            return py.inequality(model.Vmin[i], model.V[i], model.Vmax[i])
        model.vl = py.Constraint(model.B, rule=VL)

        def SUB_B (model, sub):
            return model.Th[sub] == 0.0
        model.sub_bus = py.Constraint(model.refB, rule=SUB_B)

        return model.create_instance()
    
    def Solver(self):
        Model = self.Model()
        self.opt = SolverFactory('ipopt')
        results_Model = self.opt.solve(Model)
        results_Model.write()
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        return Model

    def ExportResults(self, model):
        Flow_lines_table = pd.DataFrame(np.zeros([len(model.L),6]), columns=['from','to','Pij', 'Qij','Flow_S','Smax'])
        Power_table = pd.DataFrame(np.zeros([len(model.B),5]), columns=['bus','PG', 'QG','PL','QL'])
        Data_table = pd.DataFrame(np.zeros([len(model.B),5]), columns=['bus','Voltage', 'Angle','Gi','Bi']) 
        Ymatrix_table = pd.DataFrame(np.zeros([len(model.L),5]), columns=['from','to','Gij', 'Bij','BS'])

        r=0
        for i in model.B:
            Power_table.loc[r,'bus'] = i
            Power_table.loc[r,'PG'] = round(model.pg[i].value,4)*self.sb
            Power_table.loc[r,'QG'] = round(model.qg[i].value,4)*self.sb
            Power_table.loc[r,'PL'] = round(model.PL[i],4)
            Power_table.loc[r,'QL'] = round(model.QL[i],4)
            r += 1

        r=0
        for i in model.B:
            Data_table.loc[r,'bus'] = i
            Data_table.loc[r,'Voltage'] = round(model.V[i].value,4)
            Data_table.loc[r,'Angle'] = round(model.Th[i].value,4)
            Data_table.loc[r,'Gi'] = model.Gi[i]
            Data_table.loc[r,'Bi'] = model.Bi[i]
            r += 1

        r=0
        for (i,j) in model.L:
            Flow_lines_table.loc[r,'from'] = i
            Flow_lines_table.loc[r,'to'] = j
            Flow_lines_table.loc[r,'Pij'] = round(model.P[i,j].value,4)*self.sb
            Flow_lines_table.loc[r,'Qij'] = round(model.Q[i,j].value,4)*self.sb
            Flow_lines_table.loc[r,'Flow_S'] = round(py.sqrt((model.P[i,j].value)**2+(model.Q[i,j].value)**2)*100,4)
            Flow_lines_table.loc[r,'Smax'] = model.s_max[i,j]

            Ymatrix_table.loc[r,'from'] = i
            Ymatrix_table.loc[r,'to'] = j
            Ymatrix_table.loc[r,'Gij'] = model.Gij[i,j]
            Ymatrix_table.loc[r,'Bij'] = model.Bij[i,j]
            Ymatrix_table.loc[r,'BS'] = model.bs[i,j]
            r += 1  

        with pd.ExcelWriter('../results/results_ac_opf_nonlinear.xlsx') as data: 
            Power_table.to_excel(data, sheet_name="Power_injected")
            Data_table.to_excel(data, sheet_name="Volt_Angle")
            Flow_lines_table.to_excel(data, sheet_name="Flow_Lines")
            Ymatrix_table.to_excel(data,sheet_name='Ymatrix')

if __name__ == '__main__':
    running = ACOPF_Model()
    running.ReadExcelFile('../data/opf_33bus_distribution_system.xlsx')
    modelo = running.Solver()
    excel =  running.ExportResults(modelo)
