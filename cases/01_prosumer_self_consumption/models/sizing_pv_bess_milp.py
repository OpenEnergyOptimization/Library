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
        Cost_Inv_PV = 23.95
        Cost_Inv_BT = 35.15
        CO_PV = 0.003
        CO_BT = 0.0004

        model.Fi = Fi
        model.SOC_min = SOC_min
        model.SOC_max = SOC_max
        model.PB = PB
        model.Big_M = Big_M
        model.PL_Max = PL_Max
        model.DG = PV_Cap
        model.BT = BT
        model.Cost_Inv_PV = Cost_Inv_PV
        model.Cost_Inv_BT = Cost_Inv_BT
        model.CO_PV = CO_PV
        model.CO_BT = CO_BT

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

        model.pv_cap = Var(within = NonNegativeReals, initialize = 0)
        model.bt_cap = Var(within = NonNegativeReals, initialize = 0)


        def Fun_obj(model):
            return sum(model.Lambda[t] * (model.p_bg[t] - (model.p_sg[t]) / 3) + model.CO_PV*model.pg[t] + model.CO_BT*(model.ds[t] + model.ch[t]) for t in model.T) + model.pv_cap*model.Cost_Inv_PV + model.bt_cap*model.Cost_Inv_BT
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)


        def Gamma_R(model,t): 
            return model.pg[t] - model.PL[t] + model.ds[t] - model.ch[t] - model.p_sg[t] + model.p_bg[t] == 0
        model.gamma_r = Constraint(model.T, rule=Gamma_R)

        def Cap_Limit(model,t): 
            return model.pg[t] <= model.pv_cap * model.PG_max[t]
        model.cap_limit = Constraint(model.T, rule=Cap_Limit)

        def Cap_Limit2 (model,t): 
            return model.pv_cap <= 9
        model.cap_limit2 = Constraint(model.T, rule=Cap_Limit2)
        
        def SOCR (model,t): 
            if t == 1:
                return model.soc[t] == 0.5 * (model.bt_cap) + model.Fi * model.ch[t] - model.ds[t] / model.Fi
            else:
                return model.soc[t] == model.soc[t-1] + model.Fi*model.ch[t] - model.ds[t] / model.Fi
        model.socr = Constraint(model.T, rule=SOCR)

        def Limit_SOC1 (model,t): 
            return (model.bt_cap) * model.SOC_min <= model.soc[t]
        model.limit_soc1 = Constraint(model.T, rule=Limit_SOC1)

        def Limit_SOC2 (model,t): 
            return model.soc[t] <= model.bt_cap*model.SOC_max
        model.limit_soc2 = Constraint(model.T, rule=Limit_SOC2)

        def CHR (model,t): 
            return model.ch[t] <= (model.PB) * model.w[t]
        model.chr = Constraint(model.T, rule=CHR)

        def DSR (model,t): 
            return model.ds[t] <= (model.PB)*(1-model.w[t])
        model.dsr = Constraint(model.T, rule=DSR)

        def WR (model,t): 
            return model.w[t] <= (model.bt_cap)*1000
        model.wr = Constraint(model.T, rule=WR)

        def Limit_SOC23 (model): 
            return model.bt_cap <= 9
        model.limit_soc23 = Constraint(rule=Limit_SOC23)


        return model.create_instance()

    def Solver(self):
        Model = self.Model()
        self.opt = SolverFactory('gurobi')

        results_Model = self.opt.solve(Model)
        results_Model.write()
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        print('PV instalada: ' + str(Model.pv_cap.value))
        print('BT instalada: ' + str(Model.bt_cap.value))
        return Model
    
    def ExportResults(self, model):
        Results_Table = pd.DataFrame()
        r = 0

        for t in model.T:

            Results_Table.loc[r, 'PL']   = round(model.PL[t],   8)
            Results_Table.loc[r, 'pg']   = round(model.pg[t].value,   8)
            Results_Table.loc[r, 'p_bg'] = round(model.p_bg[t].value, 8)
            Results_Table.loc[r, 'p_sg'] = round(model.p_sg[t].value, 8)
            Results_Table.loc[r, 'soc']  = round(model.soc[t].value,  8)
            Results_Table.loc[r, 'ch']   = round(model.ch[t].value,   8)
            Results_Table.loc[r, 'ds']   = round(model.ds[t].value,   8)
            Results_Table.loc[r, 'w']    = round(model.w[t].value,    8)

            r += 1

        with pd.ExcelWriter('../results/results_sizing_pv_bess_milp.xlsx') as data:
            Results_Table.to_excel(data, sheet_name='Results', index=False)


if __name__ == '__main__':
    running = Self_Consumption_Model()
    running.ReadExcelFile('../data/prosumer_hourly_profiles_1year.xlsx')
    modelo = running.Solver()
    excel =  running.ExportResults(modelo)