from pyomo.environ import * 
import numpy as np
import pandas as pd

class DCOPF_Model:
    def __init__(self, name=None):
        self.data = {}

    def ReadExcelFile(self, FileName):
        self.Links = pd.read_excel(FileName, sheet_name='Adj_Matrix')
        self.Buses = pd.read_excel(FileName, sheet_name='Buses')
        self.Lines = pd.read_excel(FileName, sheet_name='Lines')
        self.Costs_Data = pd.read_excel(FileName, sheet_name='Costs')


        self.buses = len(self.Buses)
        self.V_min = np.unique(self.Buses['Vmin'])[0]
        self.V_max = np.unique(self.Buses['Vmax'])[0]
        self.refB = 1
        self.sb = 100
        self.Smax = self.Lines.loc[0,'Smax']*self.Links
        self.x_matrix = self.Links.copy().astype(float)

        for i in range(len(self.Lines)):
            self.fbus = self.Lines.loc[i,'fbus']
            self.tbus = self.Lines.loc[i,'tbus']
            self.aux = self.Lines.loc[i,'x']
            self.x_matrix.loc[self.fbus-1,self.tbus-1] = self.aux
            self.x_matrix.loc[self.tbus-1,self.fbus-1] = self.aux

    def Model(self):
        model = AbstractModel()


        model.B = Set(initialize=np.arange(1,self.buses+1))
        model.refB = Set(initialize=np.arange(1,self.refB+1))

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

        def PL_init(model,i):
            return self.Buses['PL'].loc[i-1]
        model.PL = Param(model.B, rule=PL_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = Param(model.B,model.B,rule=Smax_init)

        def PGmax_init(model,i):
            return self.Buses['PGmax'].loc[i-1]
        model.PGmax = Param(model.B, rule=PGmax_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.x = Param(model.B,model.B,rule=x_init)

        def Vmax_init(model,i):
            return self.Buses['Vmax'].loc[i-1]
        model.Vmax = Param(model.B, rule=Vmax_init)

        def Vmin_init(model,i):
            return self.Buses['Vmin'].loc[i-1]
        model.Vmin = Param(model.B, rule=Vmin_init)


        model.pg = Var(model.B, within = NonNegativeReals, initialize = 0)
        model.v = Var(model.B, initialize = 0)
        model.th = Var(model.B, bounds = (-8*atan(1),8*atan(1)))
        model.p = Var(model.L)


        def Fun_obj(model):
            return sum(model.Ca[i]*(model.pg[i]*self.sb)**2 + model.Cb[i]*(model.pg[i]*self.sb) + model.Cc[i] for i in model.B)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def NPFE (model,i,):
            Value = sum(model.p[k,l] for (k,l) in model.L if k==i)
            return Value == model.pg[i] - model.PL[i]/self.sb 
        model.npfe = Constraint(model.B, rule=NPFE)

        def LPFE (model,i,j):
            return model.p[i,j] == (model.th[i]-model.th[j])/model.x[i,j]
        model.lpfe = Constraint(model.L, rule=LPFE)

        def GL_PG (model, i):
            return model.pg[i] <= model.PGmax[i]/self.sb
        model.gl_pg = Constraint(model.B, rule=GL_PG)

        def V_Limit (model,i):
            return inequality(model.Vmin[i], model.v[i], model.Vmax[i])
        model.v_limit = Constraint(model.B, rule=V_Limit)

        def Smax_Limit (model,i,j):
            return  model.p[i,j] <= model.s_max[i,j]/self.sb
        model.smax_limit = Constraint(model.L, rule=Smax_Limit)

        def SUB_B (model, sub):
            return model.th[sub] == 0.0
        model.sub_bus = Constraint(model.refB, rule=SUB_B)

        return model.create_instance()

    def Solver(self):
        Model = self.Model()
        self.opt = SolverFactory('gurobi')
        results_Model = self.opt.solve(Model)
        results_Model.write()
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        return Model

    def ExportResults(self, model):
        Power_table = pd.DataFrame(columns=['bus', 'PG', 'PL'])
        Data_table = pd.DataFrame(columns=['bus', 'Voltage'])
        Flow_lines_table = pd.DataFrame(columns=['from', 'to', 'Pij'])

        r=0
        for i in model.B:
            Power_table.loc[r,'bus'] = str(i)
            Power_table.loc[r,'PG'] = round(model.pg[i].value,4)*self.sb
            Power_table.loc[r,'PL'] = round(model.PL[i],4)
            r += 1

        r=0
        for i in model.B:
            Data_table.loc[r,'bus'] = str(i)
            Data_table.loc[r,'Voltage'] = round(sqrt(model.v[i].value),4)
            r += 1

        r=0
        for (i,j) in model.L:
            Flow_lines_table.loc[r,'from'] = str(i)
            Flow_lines_table.loc[r,'to'] = str(j)
            Flow_lines_table.loc[r,'Pij'] = round(model.p[i,j].value,4)*self.sb
            r += 1  

        with pd.ExcelWriter('../results/results_dc_opf.xlsx') as data: 
            Power_table.to_excel(data, sheet_name="Power_injected")
            Data_table.to_excel(data, sheet_name="Volt_Angle")
            Flow_lines_table.to_excel(data, sheet_name="Flow_Lines")

if __name__ == '__main__':
    running = DCOPF_Model()
    running.ReadExcelFile('../data/opf_33bus_distribution_system.xlsx')
    modelo = running.Solver()
    excel =  running.ExportResults(modelo)