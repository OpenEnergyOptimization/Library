import numpy as np
import pandas as pd
import pyomo.environ as pyo


class EC_P2P_Model:
    def __init__(self, name=None):
        self.Time = 24  
        self.refB = 1
        self.Fi = 0.95
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 0.0034/2
        self.Big_M = 10
        self.Agents = [5,7,	10,	11,	12,	13,	21,	25,	26,	29,	30,	33,	34,	37,	39,	40,	42,	45,	46,	47,	49,	50,	52,	54,	57,	60,	62,	64,	66,	67,	72,	73,	75,	76,	77,	80,	81,	83,	87,	88,	93,	95,	96,	97,	102,	103,	104,	106,	107,	110,	112,	113,	114,	115,	116]

    def ReadExcelFile(self, File):
        self.Links = pd.read_excel(File, sheet_name='Adj_Matrix')
        self.Buses = pd.read_excel(File, sheet_name='Buses')
        self.Lines = pd.read_excel(File, sheet_name='Lines')
        self.Scenarios = pd.read_excel(File, sheet_name='Profiles')
        self.buses = len(self.Buses)
        self.x_matrix = pd.DataFrame(np.zeros([self.buses,self.buses]))
        self.r_matrix = pd.DataFrame(np.zeros([self.buses,self.buses]))
        self.Smax = pd.DataFrame(np.zeros([self.buses,self.buses]))

        for i in range(len(self.Lines)):
            fbus = self.Lines.loc[i,'fbus']
            tbus = self.Lines.loc[i,'tbus']
            aux = self.Lines.loc[i,'x']
            aux2 = self.Lines.loc[i,'r']
            aux3 = self.Lines.loc[i,'Smax']
            self.x_matrix.loc[fbus-1,tbus-1] = aux
            self.x_matrix.loc[tbus-1,fbus-1] = aux
            self.r_matrix.loc[fbus-1,tbus-1] = aux2
            self.r_matrix.loc[tbus-1,fbus-1] = aux2
            self.Smax.loc[fbus-1,tbus-1] = aux3
            self.Smax.loc[tbus-1,fbus-1] = aux3

    def Model(self):
        model = pyo.AbstractModel()

        model.B = pyo.Set(initialize=np.arange(1,self.buses+1))
        model.A = pyo.Set(initialize=self.Agents)
        model.refB = pyo.Set(initialize=np.arange(1,self.refB+1))
        model.T = pyo.Set(initialize=np.arange(1,self.Time+1))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1 if i<j)
        model.L = pyo.Set(dimen=2,initialize=Set_L_init)

        def PL_init(model,i,t):
            return self.Buses.loc[i-1,'PL']*self.Scenarios.loc[t-1,'P_'+str(i)]
        model.PL = pyo.Param(model.A,model.T, rule=PL_init)

        def QL_init(model,i,t):
            if i in model.A:
                return self.Buses.loc[i-1,'QL']*self.Scenarios.loc[t-1,'P_'+str(i)]
            else:
                return 0
        model.QL = pyo.Param(model.B,model.T, rule=QL_init)

        def DG_init(model,i):
            return self.Buses.loc[i-1,'DG']
        model.DG = pyo.Param(model.A, rule=DG_init)

        def BT_init(model,i):
            return self.Buses.loc[i-1,'BT']
        model.BT = pyo.Param(model.A, rule=BT_init)

        def E_BT_init(model,i):
            return self.Buses.loc[i-1,'E_BT']
        model.E_BT = pyo.Param(model.A, rule=E_BT_init)

        def PGmax_init(model,i,t):
            return self.Scenarios.loc[t-1,'PG_max']
        model.PG_max = pyo.Param(model.A,model.T, rule=PGmax_init)

        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]
        model.X = pyo.Param(model.B,model.B,rule=x_init)

        def r_init(model,i,j):
            return self.r_matrix.loc[i-1,j-1]
        model.R = pyo.Param(model.B,model.B,rule=r_init)

        def Price_init(mode,t):
            return self.Scenarios.loc[t-1,'Price']
        model.Lambda_BG = pyo.Param(model.T, rule=Price_init)

        def Price_SG_init(mode,t):
            return self.Scenarios.loc[t-1,'Price_SG']
        model.Lambda_SG = pyo.Param(model.T, rule=Price_SG_init)

        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]
        model.s_max = pyo.Param(model.B,model.B,rule=Smax_init)

        def P_SG_max_init(model,i):
            return self.Buses.loc[i-1,'P_SG_max']
        model.P_SG_max = pyo.Param(model.B, rule=P_SG_max_init)

        def P_BG_max_init(model,i):
            return self.Buses.loc[i-1,'P_BG_max']
        model.P_BG_max = pyo.Param(model.B, rule=P_BG_max_init)

        def Vmax_init(model,i):
            return self.Buses['Vmax'].loc[i-1]
        model.Vmax = pyo.Param(model.B, rule=Vmax_init)

        def Vmin_init(model,i):
            return self.Buses['Vmin'].loc[i-1]
        model.Vmin = pyo.Param(model.B,  rule=Vmin_init)

        def P_BT_max_init(model, i):
            return self.Buses.loc[i - 1, "P_BT_max"]
        model.P_BT_max = pyo.Param(model.A, rule=P_BT_max_init)


        model.Fi = pyo.Param(initialize = self.Fi)
        model.SOC_min = self.SOC_min
        model.SOC_max = self.SOC_max
        model.PB = self.PB
        model.Big_M = self.Big_M
        model.sb = 0.01
        model.MWtoKW = 1000
        model.C_PV = 0.01
        model.C_DS = 0.01
        model.SOC_init = 0.5
        model.SOC_end = 0.5
        model.epsilon_loss = 1e-5

        model.pg = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.vs = pyo.Var(model.B,model.T, initialize = 0)
        model.p = pyo.Var(model.L,model.T)
        model.q = pyo.Var(model.L,model.T)
        model.l = pyo.Var(model.L,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.p_bg = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.q_bg = pyo.Var(model.B,model.T, within = pyo.Reals, initialize = 0)
        model.p_sg = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.soc = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.ch = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.ds = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.w = pyo.Var(model.A,model.T, within = pyo.Binary, initialize = 0)
        model.kappa_bg = pyo.Var(model.B,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.kappa_sg = pyo.Var(model.B,model.T, within = pyo.NonNegativeReals, initialize = 0)

        model.p_sm = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.p_bm = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)

        model.y = pyo.Var(model.A,model.T, within = pyo.Binary, initialize = 0)
        model.dp = pyo.Var(model.A,model.T, within = pyo.Reals, initialize = 0)
        model.dp_pos = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)
        model.dp_neg = pyo.Var(model.A,model.T, within = pyo.NonNegativeReals, initialize = 0)

        model.loss_alloc = pyo.Var(model.A, model.T,within=pyo.NonNegativeReals,initialize=0)


        def Fun_obj(model):
            return  sum(model.Lambda_BG[t]*(model.kappa_bg[i,t]*model.sb) - model.Lambda_SG[t]*(model.kappa_sg[i,t]*model.sb) for i in model.B for t in model.T) + \
                    sum(model.pg[i,t]*model.sb*model.C_PV + model.ds[i,t]*model.sb*model.C_DS for i in model.A for t in model.T) +\
                    model.epsilon_loss * sum(model.R[i,j] * model.l[i,j,t] for (i,j) in model.L for t in model.T )
        model.FunObj = pyo.Objective(rule = Fun_obj, sense = pyo.minimize)

        def NPFE_P (model,i,t):
            value1 = sum(model.p[m,n,t] for (m,n) in model.L if m==i) -sum(model.p[n,m,t] - model.R[n,m]*model.l[n,m,t] for (n,m) in model.L if m==i)
            if i in model.A:
                return value1 == model.dp[i,t] 
            else:
                return value1 == model.kappa_bg[i,t]  - model.kappa_sg[i,t] 
        model.npfe_p = pyo.Constraint(model.B,model.T, rule=NPFE_P)

        def NPFE_Q (model,i,t):
            value2 = sum(model.q[m,n,t] for (m,n) in model.L if m==i) - sum(model.q[n,m,t] - model.X[n,m]*model.l[n,m,t] for (n,m) in model.L if m==i)
            return value2 == model.q_bg[i,t] - (model.QL[i,t]/model.sb) 
        model.npfe_q = pyo.Constraint(model.B,model.T, rule=NPFE_Q)

        def Gamma_R (model,i,t): 
            return model.pg[i,t] - (model.PL[i,t]/model.sb) + model.ds[i,t] - model.ch[i,t] == model.dp[i,t] 
        model.gamma_r = pyo.Constraint(model.A,model.T, rule=Gamma_R)

        def Loss_Allocation(model, t):
            return sum(model.loss_alloc[i,t] for i in model.A) == sum(model.R[i,j] * model.l[i,j,t]for (i,j) in model.L)
        model.loss_allocation = pyo.Constraint( model.T, rule=Loss_Allocation)

        def LPFE_R (model, i,j,t):
            return model.vs[j,t] == model.vs[i,t] - 2*(model.R[i,j]*model.p[i,j,t] + model.X[i,j]*model.q[i,j,t]) + (model.R[i,j]**2 + model.X[i,j]**2)*model.l[i,j,t]
        model.lpfe_r = pyo.Constraint(model.L,model.T, rule=LPFE_R)

        def LPFE_Iq (model, i,j,t):
            return (model.p[i,j,t]**2 + model.q[i,j,t]**2) <= model.l[i,j,t]*model.vs[i,t]
        model.lpfe_iq = pyo.Constraint(model.L,model.T,rule=LPFE_Iq)

        def LPFE_Iq2 (model, i,j,t):
            return (model.p[i,j,t]**2 + model.q[i,j,t]**2) <= (model.s_max[i,j]/model.sb)**2
        model.lpfe_iq2 = pyo.Constraint(model.L,model.T,rule=LPFE_Iq2)

        def Cap_Limit (model,i,t): 
            return pyo.inequality(0, model.pg[i,t], (model.DG[i]/model.sb)*model.PG_max[i,t])
        model.cap_limit = pyo.Constraint(model.A,model.T, rule=Cap_Limit)

        def P_SG_Const (model,i,t): 
            return model.kappa_sg[i,t] <= (model.P_SG_max[i]/model.sb)
        model.kappa_sg_const = pyo.Constraint(model.B, model.T, rule=P_SG_Const)

        def P_BG_Const (model,i,t): 
            return model.kappa_bg[i,t] <= (model.P_BG_max[i]/model.sb)
        model.kappa_bg_const = pyo.Constraint(model.B, model.T, rule=P_BG_Const)

        def Q_BG_Const (model,i,t): 
            return pyo.inequality(-1*(model.P_SG_max[i]/model.sb), model.q_bg[i,t], (model.P_SG_max[i]/model.sb))
        model.q_bg_const = pyo.Constraint(model.B, model.T, rule=Q_BG_Const)

        def V_Limit (model,i,t): 
            return pyo.inequality(model.Vmin[i]**2, model.vs[i,t], model.Vmax[i]**2)
        model.v_limit = pyo.Constraint(model.B,model.T, rule=V_Limit)


        def SUB_B (model, sub,t):
            return model.vs[sub,t] == 1
        model.sub_bus = pyo.Constraint(model.refB,model.T, rule=SUB_B)

        def SOCR (model,i,t): 
            if t==1:
                return model.soc[i,t] == model.SOC_init*(model.BT[i]/model.sb) + model.Fi*model.ch[i,t] - model.ds[i,t]/model.Fi
            else:
                return model.soc[i,t] == model.soc[i,t-1] + model.Fi*model.ch[i,t] - model.ds[i,t]/model.Fi
        model.socr = pyo.Constraint(model.A,model.T, rule=SOCR)

        def SOCR3 (model,i): 
            return model.soc[i,24] == model.SOC_end*(model.BT[i]/model.sb)
        model.socr3 = pyo.Constraint(model.A, rule=SOCR3)

        def Limit_SOC (model,i,t): 
            return pyo.inequality((model.BT[i]/model.sb)*model.SOC_min, model.soc[i,t],(model.BT[i]/model.sb)*model.SOC_max)
        model.limit_soc = pyo.Constraint(model.A,model.T, rule=Limit_SOC)

        def CHR (model,i,t): 
            pb = model.P_BT_max[i] / model.sb
            return model.ds[i,t] <= (pb)*(1-model.w[i,t]) - (pb)*(1-model.E_BT[i])
        model.chr = pyo.Constraint(model.A,model.T, rule=CHR)

        def DSR (model,i,t): 
            pb = model.P_BT_max[i] / model.sb
            return model.ch[i,t] <= (pb)*model.w[i,t] 
        model.dsr = pyo.Constraint(model.A,model.T, rule=DSR)

        def W_EBT (model,i,t): 
            return model.w[i,t] <= model.E_BT[i]
        model.w_ebt = pyo.Constraint(model.A,model.T,rule=W_EBT)



        def Market_Constraint_1 (model,i,t): 
            return model.dp_pos[i,t] == model.p_sg[i,t] + model.p_sm[i,t]
        model.market_constraint_1 = pyo.Constraint(model.A,model.T, rule= Market_Constraint_1)

        def Market_Constraint_2 (model,i,t): 
            return model.dp_neg[i,t] == model.p_bg[i,t] + model.p_bm[i,t]
        model.market_constraint_2 = pyo.Constraint(model.A,model.T, rule= Market_Constraint_2)

        def Market_Constraint_3 (model,i,t): 
            return model.dp[i,t]  == model.dp_pos[i,t] - model.dp_neg[i,t] +  model.loss_alloc[i,t]
        model.market_constraint_3 = pyo.Constraint(model.A,model.T, rule= Market_Constraint_3)

        def Market_Constraint_4 (model,i,t): 
            return model.dp_pos[i,t] <= model.Big_M*model.y[i,t]
        model.market_constraint_4 = pyo.Constraint(model.A,model.T, rule= Market_Constraint_4)

        def Market_Constraint_5 (model,i,t): 
            return model.dp_neg[i,t] <= model.Big_M*(1-model.y[i,t])
        model.market_constraint_5 = pyo.Constraint(model.A,model.T, rule= Market_Constraint_5)

        def Market_Constraint_7 (model,t): 
            return sum(model.p_sg[i,t] for i in model.A) == sum(model.kappa_sg[i,t] for i in model.B)
        model.market_constraint_7 = pyo.Constraint(model.T, rule= Market_Constraint_7)

        def Market_Constraint_8 (model,t): 
            return sum(model.p_bg[i,t] for i in model.A) == sum(model.kappa_bg[i,t] for i in model.B)
        model.market_constraint_8 = pyo.Constraint(model.T, rule= Market_Constraint_8)

        def Market_Constraint_9 (model,t): 
            return sum(model.p_sm[i,t] for i in model.A) == sum(model.p_bm[i,t] for i in model.A)
        model.market_constraint_9 = pyo.Constraint(model.T, rule= Market_Constraint_9)

        return model.create_instance()

    def Solver(self):
        Model = self.Model()
        self.opt = pyo.SolverFactory("gurobi")
        results = self.opt.solve(Model)
        results.write()
        return Model

    def ExportResults(self, mp):
        Lines_Columns = ['fbus', 'tbus']
        Flow_Table = pd.DataFrame(np.zeros([len(mp.L),len(Lines_Columns)]), columns=Lines_Columns)
        Loading_Table = pd.DataFrame(np.zeros([len(mp.L),len(Lines_Columns)]), columns=Lines_Columns)
        Losses_Table = pd.DataFrame(np.zeros([len(mp.L), len(Lines_Columns)]),columns=Lines_Columns)
        Voltage_Table = pd.DataFrame()
        pg_Table = pd.DataFrame()
        qg_Table = pd.DataFrame()
        p_cr_Table = pd.DataFrame()
        p_vr_Table = pd.DataFrame()
        PL_Table = pd.DataFrame()
        QL_Table = pd.DataFrame()
        soc_Table = pd.DataFrame()
        ch_Table = pd.DataFrame()
        ds_Table = pd.DataFrame()
        w_Table = pd.DataFrame()
        y1_Table = pd.DataFrame()
        p_sm_Table = pd.DataFrame()
        p_bm_Table = pd.DataFrame()
        kappa_bg_Table = pd.DataFrame()
        kappa_sg_Table = pd.DataFrame()
        Electric_Check = pd.DataFrame(np.zeros([0,2]), columns=['Bus','Variable'])

        r=0
        for (i,j) in mp.L:
            for t in mp.T:
                Flow_Table.loc[r,'fbus'] = i
                Flow_Table.loc[r,'tbus'] = j
                Flow_Table.loc[r,t] = round(mp.p[i,j,t].value,4)*mp.sb*mp.MWtoKW
                Loading_Table.loc[r,'fbus'] = i
                Loading_Table.loc[r,'tbus'] = j
                Loading_Table.loc[r,t] = pyo.sqrt((mp.p[i,j,t].value*mp.sb)**2+(mp.q[i,j,t].value*mp.sb)**2)/mp.s_max[i,j]
                Losses_Table.loc[r, 'fbus'] = i
                Losses_Table.loc[r, 'tbus'] = j
                Losses_Table.loc[r, t] = (pyo.value(mp.R[i, j]) * pyo.value(mp.l[i, j, t]) * pyo.value(mp.sb) * pyo.value(mp.MWtoKW))
            r += 1

        for i in mp.B:
            for t in mp.T:
                Voltage_Table.loc[i-1,'Bus'] = i
                Voltage_Table.loc[i-1,t] = round(pyo.sqrt(mp.vs[i,t].value),4)
                qg_Table.loc[i-1,'Bus'] = i
                qg_Table.loc[i-1,t] = round(mp.q_bg[i,t].value,4)*mp.sb*mp.MWtoKW
                QL_Table.loc[i-1,'Bus'] = i
                QL_Table.loc[i-1,t] = mp.QL[i,t]*mp.MWtoKW
                kappa_bg_Table.loc[i-1,'Bus'] = i
                kappa_bg_Table.loc[i-1,t] = round(mp.kappa_bg[i,t].value,4)*mp.sb*mp.MWtoKW
                kappa_sg_Table.loc[i-1,'Bus'] = i
                kappa_sg_Table.loc[i-1,t] = round(mp.kappa_sg[i,t].value,4)*mp.sb*mp.MWtoKW

        for i in mp.A:
            for t in mp.T:
                pg_Table.loc[i-1,'Bus'] = i
                pg_Table.loc[i-1,t] = round(mp.pg[i,t].value,4)*mp.sb*mp.MWtoKW
                p_cr_Table.loc[i-1,'Bus'] = i
                p_cr_Table.loc[i-1,t] = round(mp.p_bg[i,t].value,4)*mp.sb*mp.MWtoKW
                p_vr_Table.loc[i-1,'Bus'] = i
                p_vr_Table.loc[i-1,t] = round(mp.p_sg[i,t].value,4)*mp.sb*mp.MWtoKW
                PL_Table.loc[i-1,'Bus'] = i
                PL_Table.loc[i-1,t] = mp.PL[i,t]*mp.MWtoKW
                soc_Table.loc[i-1,'Bus'] = i
                soc_Table.loc[i-1,t] = round(mp.soc[i,t].value,4)*mp.sb*mp.MWtoKW
                ch_Table.loc[i-1,'Bus'] = i
                ch_Table.loc[i-1,t] = round(mp.ch[i,t].value,4)*mp.sb*mp.MWtoKW
                ds_Table.loc[i-1,'Bus'] = i
                ds_Table.loc[i-1,t] = round(mp.ds[i,t].value,4)*mp.sb*mp.MWtoKW
                w_Table.loc[i-1,'Bus'] = i
                w_Table.loc[i-1,t] = round(mp.w[i,t].value,4)
                y1_Table.loc[i-1,'Bus'] = i
                y1_Table.loc[i-1,t] = round(mp.y[i,t].value,4) 
                p_sm_Table.loc[i-1,'Bus'] = i
                p_sm_Table.loc[i-1,t] = round(mp.p_sm[i,t].value,4)*mp.sb*mp.MWtoKW
                p_bm_Table.loc[i-1,'Bus'] = i
                p_bm_Table.loc[i-1,t] = round(mp.p_bm[i,t].value,4)*mp.sb*mp.MWtoKW

        r=0
        for i in mp.A:
            for t in mp.T:
                Electric_Check.loc[r,'Bus'] = i
                Electric_Check.loc[r,t] = round((mp.p_bm[i,t].value - mp.p_sm[i,t].value  +  mp.pg[i,t].value + mp.p_bg[i,t].value - mp.p_sg[i,t].value - (mp.PL[i,t]/mp.sb) + mp.ds[i,t].value - mp.ch[i,t].value - mp.loss_alloc[i,t].value)*mp.sb*mp.MWtoKW,4)
            r += 1


        SOC_Gap_Abs_Table = pd.DataFrame(np.zeros([len(mp.L), len(Lines_Columns)]),columns=Lines_Columns)
        SOC_Gap_Rel_Table = pd.DataFrame(np.zeros([len(mp.L), len(Lines_Columns)]),columns=Lines_Columns)

        r = 0
        soc_gap_abs_values = []
        soc_gap_rel_values = []

        for (i, j) in mp.L:

            SOC_Gap_Abs_Table.loc[r, 'fbus'] = i
            SOC_Gap_Abs_Table.loc[r, 'tbus'] = j

            SOC_Gap_Rel_Table.loc[r, 'fbus'] = i
            SOC_Gap_Rel_Table.loc[r, 'tbus'] = j

            for t in mp.T:

                p_value = pyo.value(mp.p[i, j, t])
                q_value = pyo.value(mp.q[i, j, t])
                l_value = pyo.value(mp.l[i, j, t])
                v_value = pyo.value(mp.vs[i, t])

                cone_norm = np.sqrt((2 * p_value)**2 + (2 * q_value)**2 + (l_value - v_value)**2)

                soc_gap_abs = (l_value + v_value) - cone_norm

                denominator = max(abs(l_value + v_value), 1e-10)
                soc_gap_rel = soc_gap_abs / denominator

                SOC_Gap_Abs_Table.loc[r, t] = soc_gap_abs
                SOC_Gap_Rel_Table.loc[r, t] = soc_gap_rel

                soc_gap_abs_values.append(soc_gap_abs)
                soc_gap_rel_values.append(soc_gap_rel)

            r += 1


        max_soc_gap_abs = max(soc_gap_abs_values)
        min_soc_gap_abs = min(soc_gap_abs_values)
        mean_soc_gap_abs = np.mean(np.abs(soc_gap_abs_values))
        max_abs_soc_gap = max(abs(value) for value in soc_gap_abs_values)

        max_soc_gap_rel = max(soc_gap_rel_values)
        min_soc_gap_rel = min(soc_gap_rel_values)
        mean_soc_gap_rel = np.mean(np.abs(soc_gap_rel_values))
        max_abs_soc_gap_rel = max(abs(value) for value in soc_gap_rel_values)

        print('--- Absolute SOC relaxation gap ---')
        print('Maximum gap: ' + str(max_soc_gap_abs))
        print('Minimum gap: ' + str(min_soc_gap_abs))
        print('Mean absolute gap: ' + str(mean_soc_gap_abs))
        print('Maximum absolute gap: ' + str(max_abs_soc_gap))

        print('--- Relative SOC relaxation gap ---')
        print('Maximum relative gap: ' + str(max_soc_gap_rel))
        print('Minimum relative gap: ' + str(min_soc_gap_rel))
        print('Mean absolute relative gap: ' + str(mean_soc_gap_rel))
        print('Maximum absolute relative gap: ' + str(max_abs_soc_gap_rel))
        print('Maximum absolute relative gap (%): ' + str(100 * max_abs_soc_gap_rel))

        name = '../results/results_energy_community_p2p.xlsx'    
        with pd.ExcelWriter(name) as data:
            kappa_bg_Table.to_excel(data, sheet_name='kappa_BG')
            kappa_sg_Table.to_excel(data, sheet_name='kappa_SG')
            Electric_Check.to_excel(data, sheet_name='E_Check')
            Voltage_Table.to_excel(data, sheet_name="Voltage")
            pg_Table.to_excel(data, sheet_name="pg")
            qg_Table.to_excel(data, sheet_name="qg")
            p_cr_Table.to_excel(data, sheet_name="p_bg")
            p_vr_Table.to_excel(data, sheet_name="p_sg")
            p_sm_Table.to_excel(data, sheet_name="p_sm")
            p_bm_Table.to_excel(data, sheet_name="p_bm")
            PL_Table.to_excel(data, sheet_name="PL")
            QL_Table.to_excel(data, sheet_name="QL")
            Flow_Table.to_excel(data, sheet_name="Flow")
            Loading_Table.to_excel(data, sheet_name="Loading")
            Losses_Table.to_excel(data,sheet_name='Losses')
            soc_Table.to_excel(data, sheet_name="soc")
            ch_Table.to_excel(data, sheet_name="ch")
            ds_Table.to_excel(data, sheet_name="ds")
            w_Table.to_excel(data, sheet_name="w")
            y1_Table.to_excel(data, sheet_name="y1")
            SOC_Gap_Abs_Table.to_excel(data, sheet_name='SOC_Gap_Abs')
            SOC_Gap_Rel_Table.to_excel(data, sheet_name='SOC_Gap_Rel')
            

        print('Objective Function Result: ' + str(round(pyo.value(mp.FunObj),8))) 

if __name__ == '__main__':
    running = EC_P2P_Model()
    running.ReadExcelFile('../data/p2p_117bus_lv_reinforced_6kW.xlsx')
    modelo = running.Solver()
    excel =  running.ExportResults(modelo)