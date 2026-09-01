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
            return model.gamma_bt <= model.Gamma_PV_Max
        model.bt_limit = Constraint(rule=BT_limit)

        def W_init (model,k,t,y): 
            return model.w[k,t,y] <= model.gamma_bt*model.Big_M
        model.w_init = Constraint(model.K, model.T, model.Y, rule=W_init)

        model.balance_SP = ConstraintList()
        model.cap_limit_SP = ConstraintList()
        model.socr_SP = ConstraintList()
        model.limit_soc_SP = ConstraintList()
        model.limit_soc_2_SP = ConstraintList()
        model.chr_SP = ConstraintList()
        model.dsr_SP = ConstraintList()
        model.MP_cuts = ConstraintList()

        return model.create_instance()

    def Dual_SubProblem(self):
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
            return sum(model.PL_estimated[t,y]*model.a[t,y] + model.b[t,y]*model.Gamma_pv*model.PG_max[t,y] + model.d_minus[t,y]*model.DG_BT[y]*model.Gamma_bt*model.SOC_min + \
                        model.d_plus[t,y]*model.DG_BT[y]*model.Gamma_bt*model.SOC_max + model.e[t,y]*model.PB*model.W[t,y] + model.f[t,y]*model.PB*(1-model.W[t,y]) for t in model.T for y in model.Y) + \
                    sum(0.5*model.c[t,y]*model.Gamma_bt for t in model.T for y in model.Y if t == 1)
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
    
    def Algorithm(self):
        MP = self.Master_Problem()
        SP = self.Dual_SubProblem()

        self.opt = SolverFactory("gurobi")

        LB, UB = -np.inf, np.inf
        tol_gap = 1e-4
        GAP = 100
        limititer = 1000
        k = 0
        k_id = 1

        total_time_outer_master = 0.0
        total_time_outer_subproblem = 0.0

        MP.K.add(k_id)

        for t in MP.T:
            for y in MP.Y:
                MP.balance_SP.add(
                    MP.pg[k_id,t,y] - MP.PL_param[k_id,t,y] + MP.ds[k_id,t,y]
                    - MP.ch[k_id,t,y] - MP.p_sg[k_id,t,y] + MP.p_bg[k_id,t,y] == 0
                )
                MP.cap_limit_SP.add(MP.pg[k_id,t,y] <= MP.gamma_pv * MP.PG_max[t,y])

                if t == 1:
                    expr = MP.soc[k_id,t,y] == 0.5*MP.gamma_bt + MP.Fi*MP.ch[k_id,t,y] - MP.ds[k_id,t,y]/MP.Fi
                else:
                    expr = MP.soc[k_id,t,y] == MP.soc[k_id,t-1,y] + MP.Fi*MP.ch[k_id,t,y] - MP.ds[k_id,t,y]/MP.Fi
                MP.socr_SP.add(expr)

                MP.limit_soc_SP.add(MP.DG_BT[y]*MP.gamma_bt*MP.SOC_min <= MP.soc[k_id,t,y])
                MP.limit_soc_2_SP.add(MP.soc[k_id,t,y] <= MP.DG_BT[y]*MP.gamma_bt*MP.SOC_max)
                MP.chr_SP.add(MP.ch[k_id,t,y] <= MP.PB * MP.w[k_id,t,y])
                MP.dsr_SP.add(MP.ds[k_id,t,y] <= MP.PB * (1 - MP.w[k_id,t,y]))

        MP.MP_cuts.add(
            MP.theta_mp >= sum(
                MP.Lambda_bg[t]*MP.p_bg[k_id,t,y]
                - MP.Lambda_sg[t]*MP.p_sg[k_id,t,y]
                + MP.pg[k_id,t,y]*MP.Costo_Op_PV
                + MP.ds[k_id,t,y]*MP.Costo_Op_BT
                for t in MP.T for y in MP.Y
            )
        )

        while GAP > tol_gap:
            k += 1

            t1 = time.time()
            self.opt.solve(MP)
            t_mp = time.time() - t1
            total_time_outer_master += t_mp

            LB = value(MP.FunObj)

            SP.Gamma_pv = max(0, MP.gamma_pv.value)
            SP.Gamma_bt = max(0, MP.gamma_bt.value)

            for t in MP.T:
                for y in MP.Y:
                    SP.W[t,y] = max(0, MP.w[k_id,t,y].value)

            t2 = time.time()
            self.opt.solve(SP)
            t_sp = time.time() - t2
            total_time_outer_subproblem += t_sp

            ni = value(SP.FunObj)

            UB = min(UB, LB - MP.theta_mp.value + ni)
            GAP = abs(1 - LB/UB)

            print(f"k:{k}-----------UB:{round(UB,6)}, LB:{round(LB,6)}, GAP:{round(GAP,6)}, TIME_MP:{round(t_mp,2)}, TIME_SP:{round(t_sp,2)}")

            if GAP <= tol_gap or k >= limititer:
                break

            k_id += 1
            MP.K.add(k_id)

            for t in MP.T:
                for y in MP.Y:
                    MP.PL_param[k_id,t,y] = SP.PL_estimated[t,y].value

            MP.MP_cuts.add(
                MP.theta_mp >= sum(
                    MP.Lambda_bg[t]*MP.p_bg[k_id,t,y]
                    - MP.Lambda_sg[t]*MP.p_sg[k_id,t,y]
                    + MP.pg[k_id,t,y]*MP.Costo_Op_PV
                    + MP.ds[k_id,t,y]*MP.Costo_Op_BT
                    for t in MP.T for y in MP.Y
                )
            )

            for t in MP.T:
                for y in MP.Y:
                    MP.balance_SP.add(
                        MP.pg[k_id,t,y] - MP.PL_param[k_id,t,y] + MP.ds[k_id,t,y]
                        - MP.ch[k_id,t,y] - MP.p_sg[k_id,t,y] + MP.p_bg[k_id,t,y] == 0
                    )
                    MP.cap_limit_SP.add(MP.pg[k_id,t,y] <= MP.gamma_pv * MP.PG_max[t,y])

                    if t == 1:
                        expr = MP.soc[k_id,t,y] == 0.5*MP.gamma_bt + MP.Fi*MP.ch[k_id,t,y] - MP.ds[k_id,t,y]/MP.Fi
                    else:
                        expr = MP.soc[k_id,t,y] == MP.soc[k_id,t-1,y] + MP.Fi*MP.ch[k_id,t,y] - MP.ds[k_id,t,y]/MP.Fi
                    MP.socr_SP.add(expr)

                    MP.limit_soc_SP.add(MP.DG_BT[y]*MP.gamma_bt*MP.SOC_min <= MP.soc[k_id,t,y])
                    MP.limit_soc_2_SP.add(MP.soc[k_id,t,y] <= MP.DG_BT[y]*MP.gamma_bt*MP.SOC_max)
                    MP.chr_SP.add(MP.ch[k_id,t,y] <= MP.PB * MP.w[k_id,t,y])
                    MP.dsr_SP.add(MP.ds[k_id,t,y] <= MP.PB * (1 - MP.w[k_id,t,y]))

        print('Objective Function Result: ' + str(round(value(MP.FunObj), 8)))
        print('PV Cap: ' + str(round(MP.gamma_pv.value, 8)))
        print('BT Cap: ' + str(round(MP.gamma_bt.value, 8)))

        timing_info = {
            "ccg_outer_master_time": total_time_outer_master,
            "ccg_outer_subproblem_time": total_time_outer_subproblem,
            "ccg_total_time": total_time_outer_master + total_time_outer_subproblem,
            "ccg_iterations": k,
        }

        print(
            f"[CCG TOTAL] Iteraciones: {k} | "
            f"Master total: {total_time_outer_master:.2f} s | "
            f"Subproblem total: {total_time_outer_subproblem:.2f} s | "
            f"Total: {timing_info['ccg_total_time']:.2f} s"
        )

        return MP, SP, timing_info

    def ExportResults(self, MP, SP):
        import numpy as np
        import pandas as pd
        from pyomo.environ import value

        def f(x):
            try:
                return float(value(x))
            except:
                return np.nan

        out_file = '../results/results_robust_ders_ccg.xlsx'

        inv = pd.DataFrame([{
            "objective": f(MP.FunObj),
            "gamma_pv": f(MP.gamma_pv),
            "gamma_bt": f(MP.gamma_bt),
            "theta_mp": f(MP.theta_mp),
            "num_K": len(list(MP.K))
        }])

        sp_rows = []
        for y in SP.Y:
            for t in SP.T:
                sp_rows.append({
                    "t": int(t), "y": int(y),
                    "PL_estimated": f(SP.PL_estimated[t, y]),
                    "V_pos": int(round(f(SP.V_pos[t, y]))),
                    "V_neg": int(round(f(SP.V_neg[t, y]))),
                })
        sp_dual = pd.DataFrame(sp_rows)

        with pd.ExcelWriter(out_file) as writer:
            inv.to_excel(writer, sheet_name="INV_MP", index=False)
            sp_dual.to_excel(writer, sheet_name="SP_Dual", index=False)

            for k in sorted(list(MP.K)):
                op_rows = []
                for y in MP.Y:
                    for t in MP.T:
                        op_rows.append({
                            "k": int(k), "t": int(t), "y": int(y),
                            "PL_param": f(MP.PL_param[k, t, y]),
                            "pg": f(MP.pg[k, t, y]),
                            "p_bg": f(MP.p_bg[k, t, y]),
                            "p_sg": f(MP.p_sg[k, t, y]),
                            "soc": f(MP.soc[k, t, y]),
                            "ch": f(MP.ch[k, t, y]),
                            "ds": f(MP.ds[k, t, y]),
                            "w": int(round(f(MP.w[k, t, y]))),
                        })
                op_df = pd.DataFrame(op_rows)

                sheet = f"OP_k{int(k)}"
                op_df.to_excel(writer, sheet_name=sheet[:31], index=False)

        return out_file
    
if __name__ == '__main__':
    running = Self_Consumption_Model()
    running.ReadExcelFile('../data/robust_ders_case_study.xlsx')
    MP, SP, timing_info = running.Algorithm()
    running.ExportResults(MP, SP)
