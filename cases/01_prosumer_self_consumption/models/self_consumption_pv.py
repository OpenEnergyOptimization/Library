import numpy as np
import pandas as pd
from pyomo.environ import *

class Self_Consumption_Model:
    def __init__(self, name=None):
        self.data = {}

    def ReadExcelFile(self, FileName):
        self.data = pd.read_excel(FileName, sheet_name=None)
        self.Time = len(self.data['Profiles'])
        
    def Model(self):
        model = AbstractModel()

        model.T = Set(initialize = range(1, self.Time + 1))

        Fi = 0.95
        SOC_min = 0.1
        SOC_max = 0.9
        PB = 1
        Big_M = 1000
        PL_Max = 3.5
        PV_Cap = 6
        BT = 4

        model.Fi = Fi
        model.SOC_min = SOC_min
        model.SOC_max = SOC_max
        model.PB = PB
        model.Big_M = Big_M
        model.PL_Max = PL_Max
        model.DG = PV_Cap
        model.BT = BT

        def PL_init(model,t):
            return model.PL_Max * self.data['Profiles'].iloc[t-1, 4]
        model.PL = Param(model.T, rule=PL_init)

        def PGmax_init(model,t):
            return self.data['Profiles'].iloc[t-1, 1]
        model.PG_max = Param(model.T, rule=PGmax_init)

        def Price_init(mode,t):
            return self.data['Profiles'].iloc[t-1, 2]
        model.Lambda = Param(model.T, rule=Price_init)


        model.pg   = Var(model.T, within = NonNegativeReals, initialize = 0)
        model.p_bg = Var(model.T, within = NonNegativeReals, initialize = 0)
        model.p_sg = Var(model.T, within = NonNegativeReals, initialize = 0)
        model.soc  = Var(model.T, within = NonNegativeReals, initialize = 0)
        model.ch   = Var(model.T, within = NonNegativeReals, initialize = 0)
        model.ds   = Var(model.T, within = NonNegativeReals, initialize = 0)
        model.w    = Var(model.T, within = Binary, initialize = 0)


        def Fun_obj(model):
            return sum(model.Lambda[t] * (model.p_bg[t] - (model.p_sg[t]) / 3) for t in model.T) 
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)


        def Gamma_R(model,t): 
            return model.pg[t] - model.PL[t] - model.p_sg[t] + model.p_bg[t] == 0
        model.gamma_r = Constraint(model.T, rule=Gamma_R)

        def Cap_Limit(model,t): 
            return inequality(0, model.pg[t], model.DG * model.PG_max[t])
        model.cap_limit = Constraint(model.T, rule=Cap_Limit)


        return model.create_instance()

    def Solver(self):
        Model = self.Model()
        self.opt = SolverFactory('gurobi')

        results_Model = self.opt.solve(Model)
        results_Model.write()
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        return Model
    
    def ExportResults(self, model):
        Results_Table = pd.DataFrame()
        r = 0

        for t in model.T:

            Results_Table.loc[r, 'PL']   = round(model.PL[t],   8)
            Results_Table.loc[r, 'pg']   = round(model.pg[t].value,   8)
            Results_Table.loc[r, 'p_bg'] = round(model.p_bg[t].value, 8)
            Results_Table.loc[r, 'p_sg'] = round(model.p_sg[t].value, 8)

            r += 1

        with pd.ExcelWriter('../results/results_self_consumption_pv.xlsx') as data:
            Results_Table.to_excel(data, sheet_name='Results', index=False)


if __name__ == '__main__':
    running = Self_Consumption_Model()
    running.ReadExcelFile('../data/prosumer_hourly_profiles_1year.xlsx')
    modelo = running.Solver()
    excel =  running.ExportResults(modelo)