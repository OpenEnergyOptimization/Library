import numpy as np
import pandas as pd
from pyomo.environ import *
import time

class Self_Consumption_Model:
    def __init__(self, name=None):
        self.data = {}

    def ReadExcelFile(self, FileName):
        self.data = pd.read_excel(FileName, sheet_name=None)
        self.Profiles = pd.read_excel(FileName, sheet_name = 'Profiles')
        self.Degradation_sheet = pd.read_excel(FileName, sheet_name = 'Battery_Data')
        self.Time = 24
        self.Years = 10
        self.Costo_PV = 0.9
        self.Costo_BT = 0.8
        self.Costo_Op_PV = 0.09
        self.Costo_Op_BT = 0.08
        self.Gamma_PV_Max = 10
        self.Gamma_BT_Max = 10
        self.Budget_of_Uncertainty = 7
        
    def Original_Problem(self):
        model = AbstractModel()

        model.T = Set(initialize = range(1, self.Time + 1))
        model.Y = Set(initialize = range(1, self.Years + 1))

        self.Fi = 0.95
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 2
        self.Big_M = 1000
        self.PL_Max = 3.5

        model.Fi = self.Fi
        model.SOC_min = self.SOC_min
        model.SOC_max = self.SOC_max
        model.PB = self.PB
        model.Big_M = self.Big_M
        model.PL_Max = self.PL_Max
        model.Costo_PV = self.Costo_PV
        model.Costo_BT = self.Costo_BT
        model.Costo_Op_PV = self.Costo_Op_PV
        model.Costo_Op_BT = self.Costo_Op_BT
        model.Gamma_PV_Max = self.Gamma_PV_Max
        model.Gamma_BT_Max = self.Gamma_BT_Max

        def PL_init(model,t,y):
            return model.PL_Max * self.Profiles['P_'+str(y)].loc[t-1]
        model.PL = Param(model.T,model.Y, rule=PL_init)

        def PGmax_init(model,t,y):
            return self.Profiles['PG_max_'+str(y)].loc[t-1]
        model.PG_max = Param(model.T,model.Y, rule=PGmax_init)

        def Price_BG_init(mode,t):
            return self.Profiles['Price'].loc[t-1]
        model.Lambda_bg = Param(model.T, rule=Price_BG_init)

        def Price_SG_init(mode,t):
            return self.Profiles['Price_SG'].loc[t-1]
        model.Lambda_sg = Param(model.T, rule=Price_SG_init)

        def Degradation_BT_init(model,y):
            return self.Degradation_sheet['BT'].loc[y-1]
        model.DG_BT = Param(model.Y, rule=Degradation_BT_init)


        model.pg   = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.p_bg = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.p_sg = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.soc  = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.ch   = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.ds   = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.w    = Var(model.T,model.Y, within = Binary, initialize = 0)
        model.gamma_pv   = Var(within = NonNegativeReals, initialize = 0)
        model.gamma_bt   = Var(within = NonNegativeReals, initialize = 0)

        def Fun_obj(model):
            return sum(model.Lambda_bg[t]*model.p_bg[t,y] - model.Lambda_sg[t]*model.p_sg[t,y] + \
                    model.pg[t,y]*model.Costo_Op_PV + model.ds[t,y]*model.Costo_Op_BT for t in model.T  for y in model.Y) + \
                    model.Costo_PV*model.gamma_pv + model.Costo_BT*model.gamma_bt
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def Gamma_R(model,t,y): 
            return model.pg[t,y] - model.PL[t,y] + model.ds[t,y] - model.ch[t,y] - model.p_sg[t,y] + model.p_bg[t,y] == 0
        model.gamma_r = Constraint(model.T,model.Y, rule=Gamma_R)

        def Cap_Limit(model,t,y): 
            return model.pg[t,y] <= model.gamma_pv * model.PG_max[t,y]
        model.cap_limit = Constraint(model.T,model.Y, rule=Cap_Limit)

        def Cap_Limit_2(model): 
            return model.gamma_pv <= model.Gamma_PV_Max
        model.cap_limit_2 = Constraint( rule=Cap_Limit_2)
        
        def SOCR (model,t,y): 
            if t == 1:
                return model.soc[t,y] == 0.5*model.gamma_bt + model.Fi * model.ch[t,y] - model.ds[t,y] / model.Fi
            else:
                return model.soc[t,y] == model.soc[t-1,y] + model.Fi*model.ch[t,y] - model.ds[t,y] / model.Fi
        model.socr = Constraint(model.T,model.Y, rule=SOCR)

        def Limit_SOC (model,t,y): 
            return model.DG_BT[y]*model.gamma_bt*model.SOC_min <= model.soc[t,y]
        model.limit_soc = Constraint(model.T,model.Y, rule=Limit_SOC)

        def Limit_SOC_2 (model,t,y): 
            return  model.soc[t,y] <= model.DG_BT[y]*model.gamma_bt*model.SOC_max
        model.limit_soc_2 = Constraint(model.T,model.Y, rule=Limit_SOC_2)

        def CHR (model,t,y): 
            return model.ch[t,y] <= (model.PB) * model.w[t,y]
        model.chr = Constraint(model.T,model.Y, rule=CHR)

        def DSR (model,t,y): 
            return model.ds[t,y] <= (model.PB)*(1-model.w[t,y])
        model.dsr = Constraint(model.T,model.Y, rule=DSR)

        def BT_limit (model): 
            return model.gamma_bt <= model.Gamma_PV_Max
        model.bt_limit = Constraint(rule=BT_limit)

        return model.create_instance()

    def Master_Problem(self):
        model = AbstractModel()

        model.T = Set(initialize = range(1, self.Time + 1))
        model.Y = Set(initialize = range(1, self.Years + 1))
        model.K = Set(initialize = [])

        self.Fi = 0.95
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 2
        self.Big_M = 1000
        self.PL_Max = 3.5

        model.Fi = self.Fi
        model.SOC_min = self.SOC_min
        model.SOC_max = self.SOC_max
        model.PB = self.PB
        model.Big_M = self.Big_M
        model.PL_Max = self.PL_Max
        model.Costo_PV = self.Costo_PV
        model.Costo_BT = self.Costo_BT
        model.Costo_Op_PV = self.Costo_Op_PV
        model.Costo_Op_BT = self.Costo_Op_BT
        model.Gamma_PV_Max = self.Gamma_PV_Max
        model.Gamma_BT_Max = self.Gamma_BT_Max

        def PL_init(model,t,y):
            return model.PL_Max * self.Profiles['P_'+str(y)].loc[t-1]
        model.PL = Param(model.T,model.Y, rule=PL_init)

        def PGmax_init(model,t,y):
            return self.Profiles['PG_max_'+str(y)].loc[t-1]
        model.PG_max = Param(model.T,model.Y, rule=PGmax_init)

        def Price_BG_init(mode,t):
            return self.Profiles['Price'].loc[t-1]
        model.Lambda_bg = Param(model.T, rule=Price_BG_init)

        def Price_SG_init(mode,t):
            return self.Profiles['Price_SG'].loc[t-1]
        model.Lambda_sg = Param(model.T, rule=Price_SG_init)

        def Degradation_BT_init(model,y):
            return self.Degradation_sheet['BT'].loc[y-1]
        model.DG_BT = Param(model.Y, rule=Degradation_BT_init)


        model.pg   = Var(model.K, model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.p_bg = Var(model.K, model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.p_sg = Var(model.K, model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.soc  = Var(model.K, model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.ch   = Var(model.K, model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.ds   = Var(model.K, model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.w    = Var(model.K, model.T,model.Y, within = Binary, initialize = 0)
        model.gamma_pv   = Var(within = NonNegativeReals, initialize = 0)
        model.gamma_bt   = Var(within = NonNegativeReals, initialize = 0)
        model.theta_mp = Var(within = Reals, initialize=0, bounds=(-10000,10000))

        model.PL_param = Param(model.K, model.T, model.Y, initialize = 0, mutable=True, default=0)

        def Fun_obj(model):
            return model.Costo_PV*model.gamma_pv + model.Costo_BT*model.gamma_bt + model.theta_mp
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def Cap_Limit_2(model): 
            return model.gamma_pv <= model.Gamma_PV_Max
        model.cap_limit_2 = Constraint( rule=Cap_Limit_2)

        def BT_limit (model): 
            return model.gamma_bt <= model.Gamma_BT_Max
        model.bt_limit = Constraint(rule=BT_limit)

        model.balance_MPs = ConstraintList()
        model.cap_limit_MPs = ConstraintList()
        model.socr_MPs = ConstraintList()
        model.limit_soc_MPs = ConstraintList()
        model.limit_soc_2_MPs = ConstraintList()
        model.chr_MPs = ConstraintList()
        model.dsr_MPs = ConstraintList() 
        model.MP_cuts = ConstraintList()

        return model.create_instance()

    def Inner_Master_Problem(self):
        model = AbstractModel()

        model.T = Set(initialize = range(1, self.Time + 1))
        model.Y = Set(initialize = range(1, self.Years + 1))
        model.R = Set(initialize = []) 

        self.Fi = 0.95
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 2
        self.Big_M = 1000
        self.PL_Max = 3.5

        model.Fi = self.Fi
        model.SOC_min = self.SOC_min
        model.SOC_max = self.SOC_max
        model.PB = self.PB
        model.Big_M = self.Big_M
        model.PL_Max = self.PL_Max
        model.Costo_PV = self.Costo_PV
        model.Costo_BT = self.Costo_BT
        model.Costo_Op_PV = self.Costo_Op_PV
        model.Costo_Op_BT = self.Costo_Op_BT
        model.Gamma_PV_Max = self.Gamma_PV_Max
        model.Gamma_BT_Max = self.Gamma_BT_Max
        model.BoU = self.Budget_of_Uncertainty
        model.SOCinit = 0.5

        def PL_init(model,t,y):
            return model.PL_Max * self.Profiles['P_'+str(y)].loc[t-1]
        model.PL = Param(model.T,model.Y, rule=PL_init)

        def PL_init(model,t,y):
            return model.PL_Max * self.Profiles['P_Avg_'+str(y)].loc[t-1]
        model.PL_Barra = Param(model.T,model.Y, rule=PL_init)

        def PL_Dev_Pos_init(model,t,y):
            return model.PL_Max * self.Profiles['P_Max_'+str(y)].loc[t-1]
        model.PL_Dev_Pos = Param(model.T,model.Y, rule=PL_Dev_Pos_init)

        def PL_Dev_Neg_init(model,t,y):
            return model.PL_Max * self.Profiles['P_Min_'+str(y)].loc[t-1]
        model.PL_Dev_Neg = Param(model.T,model.Y, rule=PL_Dev_Neg_init)

        def Price_BG_init(mode,t):
            return self.Profiles['Price'].loc[t-1]
        model.Lambda_bg = Param(model.T, rule=Price_BG_init)

        def Price_SG_init(mode,t):
            return self.Profiles['Price_SG'].loc[t-1]
        model.Lambda_sg = Param(model.T, rule=Price_SG_init)

        def Degradation_BT_init(model,y):
            return self.Degradation_sheet['BT'].loc[y-1]
        model.DG_BT = Param(model.Y, rule=Degradation_BT_init)

        def PGmax_init(model,t,y):
            return self.Profiles['PG_max_'+str(y)].loc[t-1]
        model.PG_max = Param(model.T,model.Y, rule=PGmax_init)

        model.W = Param(model.T, model.Y, initialize = 0, mutable=True, default=0)
        model.Gamma_pv = Param(initialize = 0, mutable=True, default=0)
        model.Gamma_bt = Param(initialize = 0, mutable=True, default=0)

        model.a = Var(model.T, model.Y, within=Reals, initialize=0)
        model.c = Var(model.T, model.Y, within=Reals, initialize=0)
        model.b = Var(model.T, model.Y, within=NonPositiveReals, initialize=0)
        model.d_minus = Var(model.T, model.Y, within=NonNegativeReals, initialize=0)
        model.d_plus  = Var(model.T, model.Y, within=NonPositiveReals, initialize=0)
        model.e = Var(model.T, model.Y, within=NonPositiveReals,  initialize=0)
        model.f = Var(model.T, model.Y, within=NonPositiveReals,  initialize=0)
        

        model.V_pos = Var(model.T,model.Y, within = Binary, initialize = 0)
        model.V_neg = Var(model.T,model.Y, within = Binary, initialize = 0)
        model.theta = Var(within = Reals, bounds=(-10000,10000))
        model.PL_estimated = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)

        model.a_pos = Var(model.T, model.Y, within=Reals)
        model.a_neg = Var(model.T, model.Y, within=Reals)

        def Fun_obj(model):
            return model.theta
        model.FunObj = Objective(rule = Fun_obj, sense = maximize)

        def Dual_c2(model, t, y):
            return model.a[t, y] <= model.Lambda_bg[t]
        model.dual_c2 = Constraint(model.T, model.Y, rule=Dual_c2)

        def Dual_c3(model, t, y):
            return model.a[t, y] >= model.Lambda_sg[t]
        model.dual_c3 = Constraint(model.T, model.Y, rule=Dual_c3)

        def Dual_c4(model, t, y):
            return model.a[t, y] + model.b[t, y] <= model.Costo_Op_PV
        model.dual_c4 = Constraint(model.T, model.Y, rule=Dual_c4)

        def Dual_c5(model, t, y):
            return -model.a[t, y] - model.Fi * model.c[t, y] + model.e[t, y] <= 0
        model.dual_c5 = Constraint(model.T, model.Y, rule=Dual_c5)

        def Dual_c6(model, t, y):
            return model.a[t, y] + (1 / model.Fi) * model.c[t, y] + model.f[t, y] <= model.Costo_Op_BT
        model.dual_c6 = Constraint(model.T, model.Y, rule=Dual_c6)

        def Dual_c7(model, t, y):
            if t < max(model.T):
                return model.c[t, y] - model.c[t + 1, y] + model.d_minus[t, y] + model.d_plus[t, y] <= 0
            return Constraint.Skip
        model.dual_c7 = Constraint(model.T, model.Y, rule=Dual_c7)

        def Dual_c8(model, y):
            T = max(model.T)
            return model.c[T, y] + model.d_minus[T, y] + model.d_plus[T, y] <= 0
        model.dual_c8 = Constraint(model.Y, rule=Dual_c8)

        def Busget_Uncertainty_Const_PL (model,y):
            return sum(model.V_pos[t,y] + model.V_neg[t,y] for t in model.T ) <= model.BoU
        model.budget_uncertainty_const_PL = Constraint(model.Y,rule=Busget_Uncertainty_Const_PL)

        def V_constraints(model,t,y):
            return model.V_pos[t,y] + model.V_neg[t,y] <= 1
        model.v_constraints = Constraint(model.T,model.Y, rule=V_constraints)

        def PL_Estimated_Const(model,t,y):
            return model.PL_estimated[t,y] == (model.PL_Barra[t,y]) + (model.PL_Dev_Pos[t,y])*model.V_pos[t,y] - (model.PL_Dev_Neg[t,y])*model.V_neg[t,y]
        model.pl_estimated_const = Constraint(model.T,model.Y, rule=PL_Estimated_Const) 

        def Dual_A_Pos_Const_1_1(model, t, y):
            return -model.Big_M * model.V_pos[t,y] <= model.a_pos[t,y]
        model.dual_a_pos_const_1_1 = Constraint(model.T, model.Y, rule=Dual_A_Pos_Const_1_1)

        def Dual_A_Pos_Const_1_2(model, t, y):
            return model.a_pos[t,y] <= model.Big_M * model.V_pos[t,y]
        model.dual_a_pos_const_1_2 = Constraint(model.T, model.Y, rule=Dual_A_Pos_Const_1_2)

        def Dual_A_Neg_Const_1_1(model, t, y):
            return -model.Big_M * model.V_neg[t,y] <= model.a_neg[t,y]
        model.dual_a_neg_const_1_1 = Constraint(model.T, model.Y, rule=Dual_A_Neg_Const_1_1)

        def Dual_A_Neg_Const_1_2(model, t, y):
            return model.a_neg[t,y] <= model.Big_M * model.V_neg[t,y]
        model.dual_a_neg_const_1_2 = Constraint(model.T, model.Y, rule=Dual_A_Neg_Const_1_2)

        def Dual_A_Pos_Const_2_1(model, t, y):
            return model.a[t,y] - model.Big_M * (1 - model.V_pos[t,y]) <= model.a_pos[t,y]
        model.dual_a_pos_const_2_1 = Constraint(model.T, model.Y, rule=Dual_A_Pos_Const_2_1)

        def Dual_A_Pos_Const_2_2(model, t, y):
            return model.a_pos[t,y] <= model.a[t,y] + model.Big_M * (1 - model.V_pos[t,y])
        model.dual_a_pos_const_2_2 = Constraint(model.T, model.Y, rule=Dual_A_Pos_Const_2_2)

        def Dual_A_Neg_Const_2_1(model, t, y):
            return model.a[t,y] - model.Big_M * (1 - model.V_neg[t,y]) <= model.a_neg[t,y]
        model.dual_a_neg_const_2_1 = Constraint(model.T, model.Y, rule=Dual_A_Neg_Const_2_1)

        def Dual_A_Neg_Const_2_2(model, t, y):
            return model.a_neg[t,y] <= model.a[t,y] + model.Big_M * (1 - model.V_neg[t,y])
        model.dual_a_neg_const_2_2 = Constraint(model.T, model.Y, rule=Dual_A_Neg_Const_2_2)

        model.cuts = ConstraintList()

        return model.create_instance()

    def Inner_SubProblem(self):
        model = AbstractModel()

        model.T = Set(initialize = range(1, self.Time + 1))
        model.Y = Set(initialize = range(1, self.Years + 1))

        self.Fi = 0.95
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 2
        self.Big_M = 1000
        self.PL_Max = 3.5

        model.Fi = self.Fi
        model.SOC_min = self.SOC_min
        model.SOC_max = self.SOC_max
        model.PB = self.PB
        model.Big_M = self.Big_M
        model.PL_Max = self.PL_Max
        model.Costo_PV = self.Costo_PV
        model.Costo_BT = self.Costo_BT
        model.Costo_Op_PV = self.Costo_Op_PV
        model.Costo_Op_BT = self.Costo_Op_BT

        def PL_init(model,t,y):
            return model.PL_Max * self.Profiles['P_'+str(y)].loc[t-1]
        model.PL = Param(model.T,model.Y, rule=PL_init)

        def PGmax_init(model,t,y):
            return self.Profiles['PG_max_'+str(y)].loc[t-1]
        model.PG_max = Param(model.T,model.Y, rule=PGmax_init)

        def Price_BG_init(mode,t):
            return self.Profiles['Price'].loc[t-1]
        model.Lambda_bg = Param(model.T, rule=Price_BG_init)

        def Price_SG_init(mode,t):
            return self.Profiles['Price_SG'].loc[t-1]
        model.Lambda_sg = Param(model.T, rule=Price_SG_init)

        def Degradation_BT_init(model,y):
            return self.Degradation_sheet['BT'].loc[y-1]
        model.DG_BT = Param(model.Y, rule=Degradation_BT_init)

        model.pg   = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.p_bg = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.p_sg = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.soc  = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.ch   = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.ds   = Var(model.T,model.Y, within = NonNegativeReals, initialize = 0)
        model.w    = Var(model.T,model.Y, within = Binary, initialize = 0)

        model.PL_Hat = Param(model.T, model.Y, within = NonNegativeReals, initialize = 0, mutable=True)
        model.Gamma_pv = Param(initialize = 0, mutable=True, default=0)
        model.Gamma_bt = Param(initialize = 0, mutable=True, default=0)

        def Fun_obj(model):
            return sum(model.Lambda_bg[t]*model.p_bg[t,y] - model.Lambda_sg[t]*model.p_sg[t,y] + \
            model.pg[t,y]*model.Costo_Op_PV + model.ds[t,y]*model.Costo_Op_BT for t in model.T  for y in model.Y) 
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        def Gamma_R(model,t,y): 
            return model.pg[t,y] - model.PL_Hat[t,y] + model.ds[t,y] - model.ch[t,y] - model.p_sg[t,y] + model.p_bg[t,y] == 0
        model.gamma_r = Constraint(model.T,model.Y, rule=Gamma_R)

        def Cap_Limit(model,t,y): 
            return model.pg[t,y] <= model.Gamma_pv * model.PG_max[t,y]
        model.cap_limit = Constraint(model.T,model.Y, rule=Cap_Limit)
        
        def SOCR (model,t,y): 
            if t == 1:
                return model.soc[t,y] == 0.5*model.Gamma_bt + model.Fi * model.ch[t,y] - model.ds[t,y] / model.Fi
            else:
                return model.soc[t,y] == model.soc[t-1,y] + model.Fi*model.ch[t,y] - model.ds[t,y] / model.Fi
        model.socr = Constraint(model.T,model.Y, rule=SOCR)

        def Limit_SOC (model,t,y): 
            return model.DG_BT[y]*model.Gamma_bt*model.SOC_min <= model.soc[t,y]
        model.limit_soc = Constraint(model.T,model.Y, rule=Limit_SOC)

        def Limit_SOC_2 (model,t,y): 
            return  model.soc[t,y] <= model.DG_BT[y]*model.Gamma_bt*model.SOC_max
        model.limit_soc_2 = Constraint(model.T,model.Y, rule=Limit_SOC_2)

        def CHR (model,t,y): 
            return model.ch[t,y] <= (model.PB) * model.w[t,y]
        model.chr = Constraint(model.T,model.Y, rule=CHR)

        def DSR (model,t,y): 
            return model.ds[t,y] <= (model.PB)*(1-model.w[t,y])
        model.dsr = Constraint(model.T,model.Y, rule=DSR)

        return model.create_instance()

    def Solver(self):
        Model = self.Original_Problem()
        self.opt = SolverFactory('gurobi')

        results_Model = self.opt.solve(Model)
        results_Model.write()
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        print('PV Cap: ' + str(round(Model.gamma_pv.value, 8)))
        print('BT Cap: ' + str(round(Model.gamma_bt.value, 8)))
        return Model
    
    def Algorithm(self):
        MP = self.Master_Problem()
        MPs = self.Inner_Master_Problem()
        SPs = self.Inner_SubProblem()

        self.opt = SolverFactory("gurobi")

        LB, UB = -np.inf, np.inf
        tol_gap = 1e-4
        GAP = 100
        limititer = 1000
        k = 0

        total_time_outer_master = 0.0
        total_time_inner_master = 0.0
        total_time_inner_subproblem = 0.0
        total_time_outer_oracle = 0.0
        total_inner_iterations = 0

        k_id = 1
        MP.K.add(k_id)

        for t in MP.T:
            for y in MP.Y:
                MP.balance_MPs.add(
                    MP.pg[k_id, t, y] - MP.PL_param[k_id, t, y]
                    + MP.ds[k_id, t, y] - MP.ch[k_id, t, y]
                    - MP.p_sg[k_id, t, y] + MP.p_bg[k_id, t, y] == 0
                )
                MP.cap_limit_MPs.add(
                    MP.pg[k_id, t, y] <= MP.gamma_pv * MP.PG_max[t, y]
                )

                if t == 1:
                    expr = (
                        MP.soc[k_id, t, y]
                        == 0.5 * MP.gamma_bt
                        + MP.Fi * MP.ch[k_id, t, y]
                        - MP.ds[k_id, t, y] / MP.Fi
                    )
                else:
                    expr = (
                        MP.soc[k_id, t, y]
                        == MP.soc[k_id, t - 1, y]
                        + MP.Fi * MP.ch[k_id, t, y]
                        - MP.ds[k_id, t, y] / MP.Fi
                    )
                MP.socr_MPs.add(expr)

                MP.limit_soc_MPs.add(
                    MP.DG_BT[y] * MP.gamma_bt * MP.SOC_min <= MP.soc[k_id, t, y]
                )
                MP.limit_soc_2_MPs.add(
                    MP.soc[k_id, t, y] <= MP.DG_BT[y] * MP.gamma_bt * MP.SOC_max
                )
                MP.chr_MPs.add(
                    MP.ch[k_id, t, y] <= MP.PB * MP.w[k_id, t, y]
                )
                MP.dsr_MPs.add(
                    MP.ds[k_id, t, y] <= MP.PB * (1 - MP.w[k_id, t, y])
                )

        MP.MP_cuts.add(
            MP.theta_mp >= sum(
                MP.Lambda_bg[t] * MP.p_bg[k_id, t, y]
                - MP.Lambda_sg[t] * MP.p_sg[k_id, t, y]
                + MP.pg[k_id, t, y] * MP.Costo_Op_PV
                + MP.ds[k_id, t, y] * MP.Costo_Op_BT
                for t in MP.T for y in MP.Y
            )
        )

        while GAP > tol_gap:
            k += 1

            t1 = time.time()
            self.opt.solve(MP)
            t_mp = time.time() - t1
            total_time_outer_master += t_mp

            print('Gamma PV: ' + str(MP.gamma_pv.value))
            print('Gamma BT: ' + str(MP.gamma_bt.value))

            LB = value(MP.FunObj)

            MPs.Gamma_pv = MP.gamma_pv.value
            MPs.Gamma_bt = MP.gamma_bt.value
            SPs.Gamma_pv = MP.gamma_pv.value
            SPs.Gamma_bt = MP.gamma_bt.value

            LB_Inner, UB_Inner = -np.inf, np.inf
            tol_gap_inner = 1e-4
            GAP_Inner = 100
            limititer_inner = 10
            k_inner = 0

            t2 = time.time()
            print('Start Inner Problem:')

            while GAP_Inner > tol_gap_inner:
                k_inner += 1
                total_inner_iterations += 1

                t1_inner = time.time()
                self.opt.solve(MPs)
                t_mps = time.time() - t1_inner
                total_time_inner_master += t_mps

                UB_Inner = value(MPs.FunObj)

                for t in SPs.T:
                    for y in SPs.Y:
                        SPs.PL_Hat[t, y] = MPs.PL_estimated[t, y].value

                t2_inner = time.time()
                self.opt.solve(SPs)
                t_sps = time.time() - t2_inner
                total_time_inner_subproblem += t_sps

                ni_inner = value(SPs.FunObj)

                for t in MPs.T:
                    for y in MPs.Y:
                        MPs.W[t, y] = SPs.w[t, y].value

                MPs.cuts.add(
                    MPs.theta <=
                    sum(MPs.a[t, y] * MPs.PL_estimated[t, y] for t in MPs.T for y in MPs.Y)
                    + sum(MPs.c[1, y] * MPs.SOCinit * MPs.Gamma_bt for y in MPs.Y)
                    + sum(
                        MPs.b[t, y] * MPs.Gamma_pv * MPs.PG_max[t, y]
                        + MPs.d_minus[t, y] * MPs.DG_BT[y] * MPs.Gamma_bt * MPs.SOC_min
                        + MPs.d_plus[t, y] * MPs.DG_BT[y] * MPs.Gamma_bt * MPs.SOC_max
                        + MPs.e[t, y] * MPs.PB * MPs.W[t, y]
                        + MPs.f[t, y] * MPs.PB * (1 - MPs.W[t, y])
                        for t in MPs.T for y in MPs.Y
                    )
                )

                LB_Inner = max(LB_Inner, UB_Inner - MPs.theta.value + ni_inner)
                GAP_Inner = abs(1 - UB_Inner / LB_Inner)

                print(
                    "   k:%s-----------UB:%s, LB:%s, GAP:%s, TIME_MP:%s, TIME_SP:%s"
                    % (
                        k_inner,
                        round(UB_Inner, 6),
                        round(LB_Inner, 6),
                        round(GAP_Inner, 6),
                        round(t_mps, 2),
                        round(t_sps, 2),
                    )
                )

                if k_inner >= limititer_inner:
                    break

            print('End Inner Problem-------')
            t_sp = time.time() - t2
            total_time_outer_oracle += t_sp

            ni = value(MPs.FunObj)

            UB = min(UB, LB - MP.theta_mp.value + ni)
            GAP = abs(1 - LB / UB)

            print(
                "k:%s-----------UB:%s, LB:%s, GAP:%s, TIME_MP:%s, TIME_SP:%s"
                % (
                    k,
                    round(UB, 6),
                    round(LB, 6),
                    round(GAP, 6),
                    round(t_mp, 2),
                    round(t_sp, 2),
                )
            )

            if GAP <= tol_gap or k >= limititer:
                break

            k_id += 1
            MP.K.add(k_id)

            for t in MP.T:
                for y in MP.Y:
                    MP.PL_param[k_id, t, y] = MPs.PL_estimated[t, y].value

            MP.MP_cuts.add(
                MP.theta_mp >= sum(
                    MP.Lambda_bg[t] * MP.p_bg[k_id, t, y]
                    - MP.Lambda_sg[t] * MP.p_sg[k_id, t, y]
                    + MP.pg[k_id, t, y] * MP.Costo_Op_PV
                    + MP.ds[k_id, t, y] * MP.Costo_Op_BT
                    for t in MP.T for y in MP.Y
                )
            )

            for t in MP.T:
                for y in MP.Y:
                    MP.balance_MPs.add(
                        MP.pg[k_id, t, y] - MP.PL_param[k_id, t, y]
                        + MP.ds[k_id, t, y] - MP.ch[k_id, t, y]
                        - MP.p_sg[k_id, t, y] + MP.p_bg[k_id, t, y] == 0
                    )
                    MP.cap_limit_MPs.add(
                        MP.pg[k_id, t, y] <= MP.gamma_pv * MP.PG_max[t, y]
                    )

                    if t == 1:
                        expr = (
                            MP.soc[k_id, t, y]
                            == 0.5 * MP.gamma_bt
                            + MP.Fi * MP.ch[k_id, t, y]
                            - MP.ds[k_id, t, y] / MP.Fi
                        )
                    else:
                        expr = (
                            MP.soc[k_id, t, y]
                            == MP.soc[k_id, t - 1, y]
                            + MP.Fi * MP.ch[k_id, t, y]
                            - MP.ds[k_id, t, y] / MP.Fi
                        )
                    MP.socr_MPs.add(expr)

                    MP.limit_soc_MPs.add(
                        MP.DG_BT[y] * MP.gamma_bt * MP.SOC_min <= MP.soc[k_id, t, y]
                    )
                    MP.limit_soc_2_MPs.add(
                        MP.soc[k_id, t, y] <= MP.DG_BT[y] * MP.gamma_bt * MP.SOC_max
                    )
                    MP.chr_MPs.add(
                        MP.ch[k_id, t, y] <= MP.PB * MP.w[k_id, t, y]
                    )
                    MP.dsr_MPs.add(
                        MP.ds[k_id, t, y] <= MP.PB * (1 - MP.w[k_id, t, y])
                    )

        print('Objective Function Result: ' + str(round(value(MP.FunObj), 8)))
        print('PV Cap: ' + str(round(MP.gamma_pv.value, 8)))
        print('BT Cap: ' + str(round(MP.gamma_bt.value, 8)))

        timing_info = {
            "nccg_outer_master_time": total_time_outer_master,
            "nccg_inner_master_time": total_time_inner_master,
            "nccg_inner_subproblem_time": total_time_inner_subproblem,
            "nccg_outer_oracle_time": total_time_outer_oracle,
            "nccg_total_time": total_time_outer_master + total_time_outer_oracle,
            "nccg_outer_iterations": k,
            "nccg_total_inner_iterations": total_inner_iterations,
        }

        print(
            f"[NCCG TOTAL] Outer iters: {k} | Inner total iters: {total_inner_iterations} | "
            f"Outer MP total: {total_time_outer_master:.2f} s | "
            f"Inner MP total: {total_time_inner_master:.2f} s | "
            f"Inner SP total: {total_time_inner_subproblem:.2f} s | "
            f"Outer oracle total: {total_time_outer_oracle:.2f} s | "
            f"Total: {timing_info['nccg_total_time']:.2f} s"
        )

        return MP, MPs, SPs, timing_info

    def ExportResults(self, mp, mps, sps):
        Results_Table_PL = pd.DataFrame()
        Results_Table_PG = pd.DataFrame()
        Results_Table_PBG = pd.DataFrame()
        Results_Table_PSG = pd.DataFrame()
        Results_Table_SOC = pd.DataFrame()
        Results_Table_CH = pd.DataFrame()
        Results_Table_DS = pd.DataFrame()
        Results_Table_W = pd.DataFrame()
        Results_Table_INV = pd.DataFrame()

        r = 0
        for y in sps.Y:
            for t in sps.T:
                Results_Table_PL.loc[r, 'Value']   = round(mps.PL_estimated[t,y].value,   8)
                Results_Table_PL.loc[r, 't']   = t
                Results_Table_PL.loc[r, 'y']   = y
                Results_Table_PG.loc[r, 'Value']   = round(sps.pg[t,y].value,   8)
                Results_Table_PG.loc[r, 't']   = t
                Results_Table_PG.loc[r, 'y']   = y
                Results_Table_PBG.loc[r, 'Value']   = round(sps.p_bg[t,y].value,   8)
                Results_Table_PBG.loc[r, 't']   = t
                Results_Table_PBG.loc[r, 'y']   = y
                Results_Table_PSG.loc[r, 'Value']   = round(sps.p_sg[t,y].value,   8)
                Results_Table_PSG.loc[r, 't']   = t
                Results_Table_PSG.loc[r, 'y']   = y
                Results_Table_SOC.loc[r, 'Value']   = round(sps.soc[t,y].value,   8)
                Results_Table_SOC.loc[r, 't']   = t
                Results_Table_SOC.loc[r, 'y']   = y
                Results_Table_CH.loc[r, 'Value']   = round(sps.ch[t,y].value,   8)
                Results_Table_CH.loc[r, 't']   = t
                Results_Table_CH.loc[r, 'y']   = y
                Results_Table_DS.loc[r, 'Value']   = round(sps.ds[t,y].value,   8)
                Results_Table_DS.loc[r, 't']   = t
                Results_Table_DS.loc[r, 'y']   = y
                Results_Table_W.loc[r, 'Value']   = round(sps.w[t,y].value,   8)
                Results_Table_W.loc[r, 't']   = t
                Results_Table_W.loc[r, 'y']   = y
                Results_Table_INV.loc[0, 'gamma_pv']  = round(mp.gamma_pv.value, 8)
                Results_Table_INV.loc[0, 'gamma_bt']  = round(mp.gamma_bt.value, 8)
                Results_Table_INV.loc[0, 'theta_mp']  = round(mp.theta_mp.value, 8)
                Results_Table_INV.loc[0, 'objective'] = round(value(mp.FunObj), 8)

                r += 1

        with pd.ExcelWriter('../results/results_robust_ders_nccg.xlsx') as data:
            Results_Table_INV.to_excel(data, sheet_name='INV', index=False)
            Results_Table_PL.to_excel(data, sheet_name='PL', index=False)
            Results_Table_PG.to_excel(data, sheet_name='PG', index=False)
            Results_Table_PBG.to_excel(data, sheet_name='PBG', index=False)
            Results_Table_PSG.to_excel(data, sheet_name='PSG', index=False)
            Results_Table_SOC.to_excel(data, sheet_name='SOC', index=False)
            Results_Table_CH.to_excel(data, sheet_name='CH', index=False)
            Results_Table_DS.to_excel(data, sheet_name='DS', index=False)
            Results_Table_W.to_excel(data, sheet_name='W', index=False)

if __name__ == '__main__':
    running = Self_Consumption_Model()
    running.ReadExcelFile('../data/robust_ders_case_study.xlsx')
    mp, mps, sps, timing_info = running.Algorithm()
    excel =  running.ExportResults(mp, mps, sps)
