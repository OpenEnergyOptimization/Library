import numpy as np
import pandas as pd
from pyomo.environ import * 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from concurrent.futures import ProcessPoolExecutor
import time


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Select the MV test system. All case-dependent data are defined in NETWORK_CASES.
NETWORK_CASE = "IEEE69"  # Options: "IEEE33", "IEEE69"

NETWORK_CASES = {
    "IEEE33": {
        "lv_file": 'Data_206_LVDN_33MV.xlsx',
        "mv_file": 'Data_33_MVDN.xlsx',
        "boundary_buses": [3, 8, 19, 21, 24, 29],
        "agents_mv": [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 25, 26, 27, 28, 30, 31, 33],
    },
    "IEEE69": {
        "lv_file": 'Data_206_LVDN_69MV.xlsx',
        "mv_file": 'Data_69_MVDN.xlsx',
        "boundary_buses": [59, 8, 11, 12, 64, 48, 50],
        "agents_mv": [2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 65, 66, 67, 68, 69],
    },
}

# General model parameters
SB_MV = 100
SB_LV = 100
SUBSTATION_BUS = 1
TIME_PERIODS = 24
SOC_MIN = 0.10
SOC_MAX = 0.90
SOC_INITIAL = 0.50
GENERAL_LB = -100000
GENERAL_UB = 100000
CIRCUMFERENCE_SEGMENTS = 12

# LV parameters
# Retained reference list; the LV model selects active buses from PL_<boundary> > 0.
LV_AGENTS = [7, 8, 10, 11, 18, 19, 22, 25, 26, 28, 31, 32, 38, 41, 42, 45, 46, 51, 53, 57, 58, 60, 62, 74, 83, 84, 94, 95, 109, 110, 127, 130, 133, 137, 138, 142, 144, 146, 148, 151, 153, 154, 157, 158, 160, 165, 166, 169, 170, 173, 174, 177, 178, 191, 192]
PB_L = 0.0005
BIG_M_LV = 0.01
BESS_EFFICIENCY = 0.95
C_PV_LV = 0.0001
C_BT_LV = 0.0001

# MV parameters
TOTAL_PV_MV = 1
MV_BESS_UNITS = 2
PB_M_LV_LIMIT = 0.2  # MV power used in the LV market limit (original value).
PB_M = 0.02         # MV model parameter (original value).
C_PV_MV = 0.0001
C_AUX_LOAD_MV = 100

# ADMM / Lagrangian parameters
RHO = 15
ADMM_TOL_GAP = 1e-5
ADMM_MAX_ITERATIONS = 40
ADMM_INITIAL_GAP = 10
ADMM_U_STEP = 0.5
LAGRANGIAN_STEP = 0.05
ADJUSTMENT_INITIAL = 1
DUAL_INITIAL_SC = 1
DUAL_INITIAL_SE = 10
DUAL_DECIMALS = 8

# Solver and parallel execution
SOLVER_NAME = "gurobi"
LV_SOLVER_THREADS = 1
MAX_WORKERS = None  # None: one process per LV network; otherwise use this integer.

# Output configuration (console output is preserved)
RESULTS_LV_FILE = "LV_Outcomes_LV_{dnn}.xlsx"
RESULTS_MV_FILE = "MV_Outcomes_ADMM.xlsx"
RESULT_DECIMALS = 6
FLOW_RESULT_DECIMALS = 4


class Binivel_MV_LV_Problem:
    def __init__(self, name=None):
        self.Links = []
        self.Buses = []
        self.Lines = []

        if NETWORK_CASE not in NETWORK_CASES:
            raise ValueError(f"Unknown NETWORK_CASE: {NETWORK_CASE}. Use one of {list(NETWORK_CASES)}")

        self.case = NETWORK_CASES[NETWORK_CASE]
        self.File_LV = self.case["lv_file"]
        self.File_MV = self.case["mv_file"]
        self.Boundary_Buses = self.case["boundary_buses"]
        self.Agents_MV = self.case["agents_mv"]
        self.sb_MV = SB_MV
        self.sb_LV = SB_LV
        self.Substation = SUBSTATION_BUS
        self.Time = TIME_PERIODS
        self.SOC_min = SOC_MIN
        self.SOC_max = SOC_MAX
        self.SOC_initial = SOC_INITIAL
        self.LB = GENERAL_LB
        self.UB = GENERAL_UB
        self.RR = CIRCUMFERENCE_SEGMENTS
        self.Rho = RHO
        self.PB_L = PB_L
        self.BigM_LV = BIG_M_LV
        self.Fi = BESS_EFFICIENCY
        self.C_PV_LV = C_PV_LV
        self.C_BT_LV = C_BT_LV
        self.Total_PV_MV = TOTAL_PV_MV
        self.MV_BESS_units = MV_BESS_UNITS
        self.PB_M_LV_Limit = PB_M_LV_LIMIT
        self.PB_M = PB_M
        self.C_PV_MV = C_PV_MV
        self.C_Aux_Load_MV = C_AUX_LOAD_MV
        self.ADMM_tol_gap = ADMM_TOL_GAP
        self.ADMM_max_iterations = ADMM_MAX_ITERATIONS
        self.ADMM_initial_gap = ADMM_INITIAL_GAP
        self.ADMM_u_step = ADMM_U_STEP
        self.Lagrangian_step = LAGRANGIAN_STEP
        self.Adjustment_initial = ADJUSTMENT_INITIAL
        self.Dual_initial_SC = DUAL_INITIAL_SC
        self.Dual_initial_SE = DUAL_INITIAL_SE
        self.Dual_Decimals = DUAL_DECIMALS
        self.Solver_Name = SOLVER_NAME
        self.LV_Solver_Threads = LV_SOLVER_THREADS
        self.Max_Workers = MAX_WORKERS
        self.Results_LV_File = RESULTS_LV_FILE
        self.Results_MV_File = RESULTS_MV_FILE
        self.Result_Decimals = RESULT_DECIMALS
        self.Flow_Result_Decimals = FLOW_RESULT_DECIMALS

    def ReadExcelFile_MV(self, File_MV):
        self.Data_Ybus_MV = pd.read_excel(File_MV, sheet_name='input_Ybus')
        self.Lines_MV = pd.read_excel(File_MV, sheet_name='Adj_Matrix')
        self.Bus_Data_MV = pd.read_excel(File_MV, sheet_name='Buses')
        self.Branch_Data_MV = pd.read_excel(File_MV, sheet_name='Lines')
        self.Profiles_MV = pd.read_excel(File_MV, sheet_name='Profiles')
        self.Circum = pd.read_excel(File_MV, sheet_name='Circum')
        self.Costos_PG = pd.read_excel(File_MV, sheet_name='Costos')

        self.NBus_MV = len(self.Bus_Data_MV)
        self.refB_MV = int(self.Bus_Data_MV['refB'][0])

        self.X_Matrix_MV = pd.DataFrame(np.zeros([self.NBus_MV,self.NBus_MV]))
        self.R_Matrix_MV = pd.DataFrame(np.zeros([self.NBus_MV,self.NBus_MV]))
        self.Smax_MV = pd.DataFrame(np.zeros([self.NBus_MV,self.NBus_MV]))

        for i in range(len(self.Branch_Data_MV)):
            fbus = self.Branch_Data_MV.loc[i,'fbus']
            tbus = self.Branch_Data_MV.loc[i,'tbus']
            self.X_Matrix_MV.loc[fbus-1,tbus-1] = self.Branch_Data_MV.loc[i,'x']
            self.R_Matrix_MV.loc[fbus-1,tbus-1] = self.Branch_Data_MV.loc[i,'r']
            self.Smax_MV.loc[fbus-1,tbus-1] = self.Branch_Data_MV.loc[i,'Smax']

    def ReadExcelFile_LV(self, File_LV):
        self.File_LV = File_LV
        self.Data_Ybus_LV= pd.read_excel(File_LV, sheet_name='input_Ybus')
        self.Lines_LV = pd.read_excel(File_LV, sheet_name='Adj_Matrix')
        self.Bus_Data_LV = pd.read_excel(File_LV, sheet_name='Buses')
        self.Branch_Data_LV = pd.read_excel(File_LV, sheet_name='Lines')
        self.Profiles_LV = pd.read_excel(File_LV, sheet_name='Profiles')

        self.NBus_LV = len(self.Bus_Data_LV)
        self.refB_LV = int(self.Bus_Data_LV['refB'][0])

        self.Agents = LV_AGENTS

        self.X_Matrix_LV = pd.DataFrame(np.zeros([self.NBus_LV,self.NBus_LV]))
        self.R_Matrix_LV = pd.DataFrame(np.zeros([self.NBus_LV,self.NBus_LV]))
        self.X_Matrix_pu_LV = pd.DataFrame(np.zeros([self.NBus_LV,self.NBus_LV]))
        self.R_Matrix_pu_LV = pd.DataFrame(np.zeros([self.NBus_LV,self.NBus_LV]))
        self.Smax_LV = pd.DataFrame(np.zeros([self.NBus_LV,self.NBus_LV]))
        self.Lmax_LV = pd.DataFrame(np.zeros([self.NBus_LV,self.NBus_LV]))

        for i in range(len(self.Branch_Data_LV)):
            fbus = self.Branch_Data_LV.loc[i,'fbus']
            tbus = self.Branch_Data_LV.loc[i,'tbus']
            self.X_Matrix_LV.loc[fbus-1,tbus-1] = self.Branch_Data_LV.loc[i,'x']
            self.R_Matrix_LV.loc[fbus-1,tbus-1] = self.Branch_Data_LV.loc[i,'r']
            self.Smax_LV.loc[fbus-1,tbus-1] = self.Branch_Data_LV.loc[i,'Smax']
            # self.Lmax_LV.loc[fbus-1,tbus-1] = self.Branch_Data_LV.loc[i,'Imax']

    def LV_DN_Model (self,dnn): 
        model = AbstractModel()

        self.agents_testing = self.Bus_Data_LV[self.Bus_Data_LV['PL_'+str(dnn)] > 0].index.tolist()

        ## -- SETs -- ##
        model.Omega_L = Set(initialize=[(dn, i) for dn in np.array([dnn]) for i in np.arange(1,self.NBus_LV+1)])
        model.Omega_S = Set(initialize=[(dn, i) for dn in np.array([dnn]) for i in np.arange(self.Substation,self.Substation+1)])
        model.A = Set(initialize=[(dn, i+1) for dn in np.array([dnn]) for i in self.agents_testing])
        model.T_L = Set(initialize=np.arange(1,self.Time+1))

        def Set_L_init(model):
            return ((dn, i+1,j+1) for dn in np.array([dnn]) for i in np.arange(0,self.NBus_LV) for j in np.arange(0,self.NBus_LV) if self.Lines_LV.loc[i,j]==1 if i<j)
        model.L_L = Set(dimen=3,initialize=Set_L_init)
            
        ## -- PARAMETERS -- ##
        model.PGmin = Param(model.Omega_L, initialize = 0)

        def Lambda_L_BC_init(model,t):
            return self.Profiles_LV['Lambda_L_BC'].loc[t-1]
        model.Lambda_L_BC = Param(model.T_L, rule=Lambda_L_BC_init)

        def Lambda_L_BE_init(model,t):
            return self.Profiles_LV['Lambda_L_BE'].loc[t-1]
        model.Lambda_L_BE = Param(model.T_L, rule=Lambda_L_BE_init)

        def Lambda_L_SG_init(model,t):
            return self.Profiles_LV['Lambda_L_SG'].loc[t-1]
        model.Lambda_L_S = Param(model.T_L, rule=Lambda_L_SG_init)

        def Gamma_BT_init(model,bd,i):
            return self.Bus_Data_LV['Gamma_BT_'+str(bd)].loc[i-1]
        model.Gamma_BT_L = Param(model.A, rule=Gamma_BT_init)

        def BT_Loc_init(model,bd,i):
            return self.Bus_Data_LV['BT_Loc_'+str(bd)].loc[i-1]
        model.BT_Loc_L = Param(model.A, rule=BT_Loc_init)

        def PL_init(model,bd,i,t):
            return self.Bus_Data_LV['PL_'+str(bd)].loc[i-1]*self.Profiles_LV['P_'+str(i)].loc[t-1]
        model.PL = Param(model.A,model.T_L, rule=PL_init)

        def QL_init(model,bd,i,t):
            if i in model.A:
                return self.Bus_Data_LV['QL_'+str(bd)].loc[i-1]*self.Profiles_LV['P_'+str(i)].loc[t-1]
            else:
                return 0
        model.QL = Param(model.Omega_L,model.T_L, rule=QL_init)

        def PVmax_init(model,bd,i,t):
            return (self.Bus_Data_LV['DG_'+str(bd)].loc[i-1]/self.sb_LV)*self.Profiles_LV['PG_max'].loc[t-1]
        model.PVmax = Param(model.A,model.T_L, rule=PVmax_init)

        def Irradiance_init(model,t):
            return self.Profiles_LV['PG_max'].loc[t-1]
        model.Irradiance = Param(model.T_L, rule=Irradiance_init)

        def QGmin_init(model,bd,i):
            return self.Bus_Data_LV['QGmin'].loc[i-1]
        model.QGmin = Param(model.Omega_L,rule=QGmin_init)

        def QGmax_init(model,bd,i):
            return self.Bus_Data_LV['QGmax'].loc[i-1]
        model.QGmax = Param(model.Omega_L, rule=QGmax_init)

        def Vmax_init(model,bd,i):
            return self.Bus_Data_LV['Vmax'].loc[i-1]
        model.Vmax = Param(model.Omega_L, rule=Vmax_init)

        def Vmin_init(model,bd,i):
            return self.Bus_Data_LV['Vmin'].loc[i-1]
        model.Vmin = Param(model.Omega_L, rule=Vmin_init)

        def Smax_init(model,bd,i,j):
            return self.Smax_LV.loc[i-1,j-1]
        model.s_max = Param(model.L_L,rule=Smax_init)

        def X_init(model,bd,i,j):
            return self.X_Matrix_LV.loc[i-1,j-1]
        model.X = Param(model.L_L,rule=X_init)

        def R_init(model,bd,i,j):
            return self.R_Matrix_LV.loc[i-1,j-1]
        model.R = Param(model.L_L,rule=R_init)

        model.sb = Param(initialize = self.sb_LV)
        model.SOC_min_L = self.SOC_min
        model.SOC_max_L = self.SOC_max
        model.PB_L = self.PB_L
        model.Big_M = self.BigM_LV
        model.Fi = self.Fi
        model.C_PV = self.C_PV_LV
        model.C_BT = self.C_BT_LV
        model.Rho = self.Rho
        model.Total_PV_MV = self.Total_PV_MV
        model.PB_M = self.PB_M_LV_Limit

        ## -- VARIABLES -- ##
        model.pv = Var(model.A,model.T_L, within = NonNegativeReals, initialize = 0)
        model.qg = Var(model.Omega_L,model.T_L,initialize = 0)
        model.V = Var(model.Omega_L,model.T_L, initialize = 0)
        model.P = Var(model.L_L,model.T_L, initialize = 1)
        model.Q = Var(model.L_L,model.T_L, initialize = 0)
        model.l = Var(model.L_L,model.T_L)
        model.soc = Var(model.A,model.T_L, within = NonNegativeReals, initialize = 0)
        model.ds = Var(model.A,model.T_L, within = NonNegativeReals, initialize = 0)
        model.ch = Var(model.A,model.T_L, within = NonNegativeReals, initialize = 0)
        model.w = Var(model.A,model.T_L, within = Binary, initialize = 0)

        model.Lpbc = Var(model.Omega_S,model.T_L, within = NonNegativeReals, initialize = 0)
        model.Lpbe = Var(model.Omega_S,model.T_L, within = NonNegativeReals, initialize = 0)
        model.Lps = Var(model.Omega_S,model.T_L, within = NonNegativeReals, initialize = 0)

        model.dp = Var(model.A,model.T_L, within = Reals, initialize = 0)
        model.db = Var(model.Omega_S,model.T_L, within = Reals, initialize = 0)

        ## ADMM parameters
        model.Mpb_param = Param(model.Omega_S,model.T_L, initialize = 0, mutable=True)
        model.Mpsc_param = Param(model.Omega_S,model.T_L, initialize = 0, mutable=True)
        model.Mpse_param = Param(model.Omega_S,model.T_L, initialize = 0, mutable=True)

        model.u_pb_param = Param(model.Omega_S,model.T_L, initialize = 0, mutable=True)
        model.u_psc_param = Param(model.Omega_S,model.T_L, initialize = 0, mutable=True)
        model.u_pse_param = Param(model.Omega_S,model.T_L, initialize = 0, mutable=True)

        model.Ajuste_param = Param(model.Omega_S,model.T_L, initialize = self.Adjustment_initial, mutable=True)
        model.Lambda_L_BC_Dual = Param(model.Omega_S,model.T_L, initialize = self.Dual_initial_SC, mutable=True)
        model.Lambda_L_BE_Dual = Param(model.Omega_S,model.T_L, initialize = self.Dual_initial_SE, mutable=True)


        ## -- OBJECTIVE FUNCTION -- ##
        def Fun_obj(model):
            return sum(model.Lambda_L_BC[t]*model.Lpbc[bd,i,t]*model.sb + model.Lambda_L_BE[t]*model.Lpbe[bd,i,t]*model.sb - model.Lambda_L_S[t]*model.Lps[bd,i,t]*model.sb for (bd,i) in model.Omega_S for t in model.T_L) + \
                   sum(model.C_PV*model.pv[bd,i,t]*model.sb + model.C_BT*model.ds[bd,i,t]*model.sb for (bd,i) in model.A for t in model.T_L) + \
                   model.Rho/2*sum( (model.Lpbc[bd,i,t]*model.sb - model.Mpsc_param[bd,i,t]*model.sb + model.u_psc_param[bd,i,t]*model.sb)**2  for (bd,i) in model.Omega_S for t in model.T_L)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        # -- CONSTRAINTS -- ##

        def NPFE_R (model,bd,i,t):
            value1 = sum(model.P[bdd,m,n,t] for (bdd,m,n) in model.L_L if m==i if bdd==bd) - sum(model.P[bdd,n,m,t] - model.R[bdd,n,m]*model.l[bdd,n,m,t] for (bdd,n,m) in model.L_L if m==i if bdd==bd) 
            if (bd,i) in model.A:
                return model.dp[bd,i,t] == value1 
            if (bd,i) in model.Omega_S:
                return model.db[bd,i,t]   == value1 
            else:
                return 0 == value1
        model.npfe_r = Constraint(model.Omega_L,model.T_L, rule=NPFE_R)

        def NPFE_Im (model,bd,i,t):
            value2 = sum(model.Q[bdd,m,n,t] for (bdd,m,n) in model.L_L if m==i if bdd==bd) - sum(model.Q[bdd,n,m,t] - model.X[bdd,n,m]*model.l[bdd,n,m,t] for (bdd,n,m) in model.L_L if m==i if bdd==bd) 
            return model.qg[bd,i,t] - model.QL[bd,i,t]/self.sb_LV == value2
        model.npfe_im = Constraint(model.Omega_L,model.T_L, rule=NPFE_Im)

        def Market_Constraint_0 (model,bd,i,t): 
            return model.dp[bd,i,t] == model.pv[bd,i,t] - model.PL[bd,i,t]/self.sb_LV + model.ds[bd,i,t] - model.ch[bd,i,t] 
        model.market_constraint_0 = Constraint(model.A,model.T_L, rule= Market_Constraint_0)

        def Market_Constraint_1 (model,bd,i,t): 
            return model.db[bd,i,t] == model.Lpbc[bd,i,t] + model.Lpbe[bd,i,t] - model.Lps[bd,i,t]
        model.market_constraint_1 = Constraint(model.Omega_S,model.T_L, rule= Market_Constraint_1)

        def Market_Constraint_2 (model,bd,i,t): 
            return model.Lps[bd,i,t] <= sum(model.pv[bdd,j,t] + model.ds[bdd,j,t] for (bdd,j) in model.A if bdd==bd) 
        model.market_constraint_2 = Constraint(model.Omega_S,model.T_L, rule= Market_Constraint_2)

        def Market_Constraint_3 (model,bd,i,t): 
            return model.Lpbc[bd,i,t] + model.Lpbe[bd,i,t] <= sum(model.PL[bdd,j,t]/self.sb_LV + model.ch[bdd,j,t] for (bdd,j) in model.A if bdd==bd) 
        model.market_constraint_3 = Constraint(model.Omega_S,model.T_L, rule= Market_Constraint_3)

        def Market_Constraint_4 (model,bd,i,t): 
            return model.Lpbc[bd,i,t]  <= (model.Total_PV_MV/self.sb_LV*model.Irradiance[t] + self.MV_BESS_units*model.PB_M/self.sb_LV) 
        model.market_constraint_4 = Constraint(model.Omega_S,model.T_L, rule= Market_Constraint_4)

        def LPFE_R (model, bd,i,j,t):
            return model.V[bd,j,t] == model.V[bd,i,t] - 2*(model.R[bd,i,j]*model.P[bd,i,j,t] + model.X[bd,i,j]*model.Q[bd,i,j,t]) + (model.R[bd,i,j]**2 + model.X[bd,i,j]**2)*model.l[bd,i,j,t]
        model.lpfe_r = Constraint(model.L_L,model.T_L, rule=LPFE_R)

        def LPFE_Iq (model, bd,i,j,t):
            return (model.P[bd,i,j,t]**2 + model.Q[bd,i,j,t]**2) <= model.l[bd,i,j,t]*model.V[bd,i,t]
        model.lpfe_iq = Constraint(model.L_L,model.T_L, rule=LPFE_Iq)

        def GL_QG (model, bd,i,t):
            return inequality(model.QGmin[bd,i]/self.sb_LV, model.qg[bd,i,t],model.QGmax[bd,i]/self.sb_LV)
        model.gl_qg = Constraint(model.Omega_L,model.T_L, rule=GL_QG)

        def VL (model,bd,i,t):
            return inequality(model.Vmin[bd,i]**2, model.V[bd,i,t], model.Vmax[bd,i]**2)
        model.vl = Constraint(model.Omega_L,model.T_L, rule=VL)

        def GL_PG (model, bd,i,t):
            return model.pv[bd,i,t] <= model.PVmax[bd,i,t]
        model.gl_pg = Constraint(model.A,model.T_L, rule=GL_PG)

        def SOC_L (model,bd,i,t):
            if t == 1:
                return model.soc[bd,i,t] == (model.Gamma_BT_L[bd,i]/self.sb_LV)*self.SOC_initial 
            else:
                return model.soc[bd,i,t] == model.soc[bd,i,t-1] + model.Fi*model.ch[bd,i,t] - model.ds[bd,i,t]/model.Fi
        model.soc_L_Const = Constraint(model.A,model.T_L, rule=SOC_L)

        def SOC_L_Limit (model,bd,i,t):
            return inequality((model.Gamma_BT_L[bd,i]/self.sb_LV)*model.SOC_min_L, model.soc[bd,i,t],(model.Gamma_BT_L[bd,i]/self.sb_LV)*model.SOC_max_L)
        model.soc_L_Limit_Const = Constraint(model.A,model.T_L, rule=SOC_L_Limit)

        def SOCR2 (model,bd,i,t): 
            return model.ds[bd,i,t] + model.SOC_min_L*(model.Gamma_BT_L[bd,i]/self.sb_LV) <= model.soc[bd,i,t]
        model.socr2_l = Constraint(model.A,model.T_L, rule=SOCR2)

        def DS_L_Limit (model,bd,i,t):
            return model.ds[bd,i,t] <= (model.PB_L/self.sb_LV)*model.w[bd,i,t]
        model.ds_L_Limit_Const = Constraint(model.A,model.T_L, rule=DS_L_Limit)

        def CH_L_Limit (model,bd,i,t):
            return model.ch[bd,i,t] <= (model.PB_L/self.sb_LV)*(1-model.w[bd,i,t]) - (model.PB_L/self.sb_LV)*(1-model.BT_Loc_L[bd,i])
        model.ch_L_Limit_Const = Constraint(model.A,model.T_L, rule=CH_L_Limit)

        def W_L_Limit (model,bd,i,t):
            return model.w[bd,i,t] <= model.BT_Loc_L[bd,i]
        model.w_L_Limit_Const = Constraint(model.A,model.T_L, rule=W_L_Limit)        

        return model.create_instance()
    
    def MV_DN_Model (self): 
        model = AbstractModel()

        ## -- SETs -- ##
        model.Omega_M = Set(initialize=np.arange(1,self.NBus_MV+1))
        model.Omega_B = Set(initialize=self.Boundary_Buses)
        model.Omega_S = Set(initialize=np.arange(self.Substation,self.Substation+1))
        model.T_M = Set(initialize=np.arange(1,self.Time+1))
        model.A_M = Set(initialize=self.Agents_MV)
        model.RR = Set(initialize=np.arange(self.RR))

        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.NBus_MV) for j in np.arange(0,self.NBus_MV) if self.Lines_MV.loc[i,j]==1 if i<j)
        model.L_M = Set(dimen=2,initialize=Set_L_init)
            
        ## -- PARAMETERS -- ##
        model.PGmin = Param(model.Omega_M, initialize = 0)

        def AA_init(model,i):
            return self.Circum.loc[i,'AA']
        model.AA = Param(model.RR, rule=AA_init)

        def BB_init(model,i):
            return self.Circum.loc[i,'BB']
        model.BB = Param(model.RR, rule=BB_init)

        def CC_init(model,i):
            return self.Circum.loc[i,'CC']
        model.CC = Param(model.RR, rule=CC_init)

        def PL_init(model,i,t):
            return self.Bus_Data_MV['PL'].loc[i-1]*self.Profiles_MV['P_'+str(i)].loc[t-1]
        model.PL = Param(model.A_M,model.T_M, rule=PL_init)

        def QL_init(model,i,t):
            return self.Bus_Data_MV['QL'].loc[i-1]*self.Profiles_MV['P_'+str(i)].loc[t-1]
        model.QL = Param(model.A_M,model.T_M, rule=QL_init)

        def PVmax_init(model,i,t):
            return self.Bus_Data_MV['PVmax'].loc[i-1]*self.Profiles_MV['PV_max'].loc[t-1]
        model.PVmax = Param(model.A_M,model.T_M, rule=PVmax_init)

        def PGmax_init(model,i,t):
            return self.Bus_Data_MV['PGmax'].loc[i-1]
        model.PGmax = Param(model.A_M,model.T_M, rule=PGmax_init)

        def QGmin_init(model,i):
            return self.Bus_Data_MV['QGmin'].loc[i-1]
        model.QGmin = Param(model.Omega_M,rule=QGmin_init)

        def QGmax_init(model,i):
            return self.Bus_Data_MV['QGmax'].loc[i-1]
        model.QGmax = Param(model.Omega_M, rule=QGmax_init)

        def Vmax_init(model,i):
            return self.Bus_Data_MV['Vmax'].loc[i-1]
        model.Vmax = Param(model.Omega_M, rule=Vmax_init)

        def Vmin_init(model,i):
            return self.Bus_Data_MV['Vmin'].loc[i-1]
        model.Vmin = Param(model.Omega_M, rule=Vmin_init)

        def Smax_init(model,i,j):
            return self.Smax_MV.loc[i-1,j-1]
        model.s_max = Param(model.Omega_M,model.Omega_M,rule=Smax_init)

        def X_init(model,i,j):
            return self.X_Matrix_MV.loc[i-1,j-1]
        model.X = Param(model.Omega_M,model.Omega_M,rule=X_init)

        def R_init(model,i,j):
            return self.R_Matrix_MV.loc[i-1,j-1]
        model.R = Param(model.Omega_M,model.Omega_M,rule=R_init)

        def Ca_init_T(model,i):
            return self.Costos_PG['a'].loc[i-1]
        model.Ca = Param(model.A_M, rule=Ca_init_T)

        def Cb_init_T(model,i):
            return self.Costos_PG['b'].loc[i-1]
        model.Cb = Param(model.A_M, rule=Cb_init_T)

        def Cc_init_T(model,i):
            return self.Costos_PG['c'].loc[i-1]
        model.Cc = Param(model.A_M, rule=Cc_init_T)

        model.sb = Param(initialize = self.sb_MV)
        model.Rho = self.Rho
        model.SOC_min_M = self.SOC_min
        model.SOC_max_M = self.SOC_max
        model.PB_M = self.PB_M

        ## -- VARIABLES -- ##
        model.pg = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.pv_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.qg = Var(model.Omega_M,model.T_M,initialize = 0)
        model.V = Var(model.Omega_M,model.T_M, initialize = 0)
        model.P = Var(model.L_M,model.T_M, initialize = 1)
        model.Q = Var(model.L_M,model.T_M, initialize = 0)
        model.l = Var(model.L_M,model.T_M)

        model.Mpb = Var(model.Omega_B,model.T_M, within = NonNegativeReals, initialize = 0)
        model.Mpsc = Var(model.Omega_B,model.T_M, within = NonNegativeReals, initialize = 0)
        model.Mpse = Var(model.Omega_B,model.T_M, within = NonNegativeReals, initialize = 0)

        model.db_M = Var(model.Omega_B,model.T_M, within = Reals, initialize = 0)
        model.dp_M = Var(model.A_M,model.T_M, within = Reals, initialize = 0)

        model.soc_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.ds_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.ch_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.w_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0, bounds=(0,1))

        ## ADMM parameters
        model.Lpbc_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)
        model.Lpbe_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)
        model.Lps_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)

        model.u_psc_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)

        model.lambda_pb_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)
        model.lambda_pse_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)

        model.db_M_param = Param(model.Omega_B,model.T_M, initialize = 0, mutable=True)

        model.Lambda_L_SC_Dual = Param(model.Omega_B,model.T_M, initialize = self.Dual_initial_SC, mutable=True)
        model.Lambda_L_SE_Dual = Param(model.Omega_B,model.T_M, initialize = self.Dual_initial_SE, mutable=True)

        model.aux_load = Var(model.Omega_M,model.T_M, within = NonNegativeReals, initialize = 0)

        ## -- OBJECTIVE FUNCTION -- ##
        def Fun_obj(model):
            return sum(model.Ca[i]*(model.pg[i,t]*self.sb_MV)**2 + model.Cb[i]*(model.pg[i,t]*self.sb_MV) + model.Cc[i] for i in model.A_M for t in model.T_M)  + \
                   sum(model.pv_M[i,t]*self.sb_MV*self.C_PV_MV  for i in model.A_M for t in model.T_M) +\
                   model.Rho/2*sum( (model.Mpsc[i,t]*self.sb_MV - model.Lpbc_param[i,t]*self.sb_MV - model.u_psc_param[i,t]*self.sb_MV)**2 for i in model.Omega_B for t in model.T_M) + \
                   sum( model.lambda_pb_param[i,t]*(model.Mpb[i,t] - model.Lps_param[i,t]) for i in model.Omega_B for t in model.T_M  ) +\
                   sum(model.aux_load[i,t]*self.sb_MV*self.C_Aux_Load_MV  for i in model.A_M for t in model.T_M)
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)
        
        # -- CONSTRAINTS -- ##
        def NPFE_R (model,i,t):
            value1 = sum(model.P[m,n,t] for (m,n) in model.L_M if m==i) - sum(model.P[n,m,t] for (n,m) in model.L_M if m==i) 
            if i == self.Substation:
                return - model.aux_load[i,t]  == value1
            if i in model.Omega_B:
                return model.db_M[i,t]  == value1 
            if i in model.A_M:
                return model.dp_M[i,t]  == value1 
            else:
                return 0 == value1
        model.npfe_r = Constraint(model.Omega_M,model.T_M, rule=NPFE_R)

        def NPFE_Im (model,i,t):
            value2 = sum(model.Q[m,n,t] for (m,n) in model.L_M if m==i) - sum(model.Q[n,m,t] for (n,m) in model.L_M if m==i) 
            if i in model.A_M:
                return model.qg[i,t] - model.QL[i,t]/self.sb_MV == value2
            else:
                return model.qg[i,t] == value2
        model.npfe_im = Constraint(model.Omega_M,model.T_M, rule=NPFE_Im)

        def DP_Const (model, i,t):
            return model.dp_M[i,t] == model.pg[i,t] + model.pv_M[i,t] - model.PL[i,t]/self.sb_MV 
        model.dp_const = Constraint(model.A_M,model.T_M, rule=DP_Const)

        def DB_Const (model, i,t):
            return model.db_M[i,t] == model.Mpb[i,t] - model.Mpsc[i,t] - model.Mpse[i,t]
        model.db_const = Constraint(model.Omega_B,model.T_M, rule=DB_Const)

        def DB_Const_Param (model, i,t):
            return model.db_M[i,t] == -model.db_M_param[i,t]
        model.db_const_param = Constraint(model.Omega_B,model.T_M, rule=DB_Const_Param)

        def DB_Const_Param2 (model, i,t):
            return model.Mpb[i,t] <= model.Lps_param[i,t]
        model.db_const_param2 = Constraint(model.Omega_B,model.T_M, rule=DB_Const_Param2)

        def LPFE_R (model, i,j,t):
            return model.V[j,t] == model.V[i,t] - 2*(model.R[i,j]*model.P[i,j,t] + model.X[i,j]*model.Q[i,j,t]) 
        model.lpfe_r = Constraint(model.L_M,model.T_M, rule=LPFE_R)

        def GL_PG (model, i,t):
            return model.pg[i,t] <= model.PGmax[i,t]/self.sb_MV
        model.gl_pg = Constraint(model.A_M,model.T_M, rule=GL_PG)

        def GL_PV (model, i,t):
            return model.pv_M[i,t] <= model.PVmax[i,t]/self.sb_MV
        model.gl_pv = Constraint(model.A_M,model.T_M, rule=GL_PV)

        def GL_QG (model, i,t):
            return inequality(model.QGmin[i]/self.sb_MV, model.qg[i,t],model.QGmax[i]/self.sb_MV)
        model.gl_qg = Constraint(model.Omega_M,model.T_M, rule=GL_QG)

        def VL (model,i,t):
            return inequality(model.Vmin[i]**2, model.V[i,t], model.Vmax[i]**2)
        model.vl = Constraint(model.Omega_M,model.T_M, rule=VL)

        # Additional Constraints
        def PSE_PG(model,t):
            return sum(model.Mpse[i,t] for i in model.Omega_B) <= sum(model.pg[i,t] for i in model.A_M)
        model.pse_pg = Constraint(model.T_M, rule = PSE_PG)

        def PSC_PV(model,t):
            return sum(model.Mpsc[i,t] for i in model.Omega_B) <= sum(model.pv_M[i,t]  for i in model.A_M) 
        model.psc_pv = Constraint(model.T_M, rule = PSC_PV)

        return model.create_instance()
    
    def Algorithm(self):
        lv = {}
        mv = self.MV_DN_Model()
        mv.dual =Suffix(direction=Suffix.IMPORT)

        for dnn in self.Boundary_Buses:
            lv[dnn] = self.LV_DN_Model(dnn)

        LB, UB    = -np.inf, np.inf 
        tol_gap   = self.ADMM_tol_gap
        limititer = self.ADMM_max_iterations
        k = 0
        GAP = self.ADMM_initial_gap
        self.opt = SolverFactory(self.Solver_Name)
        eta = self.Lagrangian_step

        start_time = time.perf_counter()
        while GAP>tol_gap:
            k += 1

            if k > 1:  
                for dnn in self.Boundary_Buses:
                    for (bd,i) in lv[dnn].Omega_S:
                        for t in lv[dnn].T_L:
                            lv[dnn].Mpsc_param[bd,i,t] = mv.Mpsc[dnn,t].value
                            lv[dnn].u_psc_param[bd,i,t] = mv.u_psc_param[dnn,t]

            payloads = []
            for dnn in self.Boundary_Buses:
                mpsc_values = {}
                u_values = {}
                for (bd,i) in lv[dnn].Omega_S:
                    for t in lv[dnn].T_L:
                        mpsc_values[(bd,i,t)] = value(lv[dnn].Mpsc_param[bd,i,t])
                        u_values[(bd,i,t)] = value(lv[dnn].u_psc_param[bd,i,t])
                payloads.append((dnn, self.File_LV, self.Rho, mpsc_values, u_values))

            with ProcessPoolExecutor(max_workers=self.Max_Workers if self.Max_Workers is not None else len(self.Boundary_Buses)) as executor:
                results = list(executor.map(solve_lv_process, payloads))

            for dnn, obj_val, solution in results:
                for var_name, values in solution.items():
                    var_component = getattr(lv[dnn], var_name)
                    for idx, val in values.items():
                        if var_component[idx].is_binary():
                            val = round(val)
                        var_component[idx].set_value(val)

            FO_LV_sum_1 = sum(value(lv[dnn].FunObj) for dnn in self.Boundary_Buses)
            Penalizater = self.Rho/2*sum( (lv[dnn].Lpbc[bd,i,t].value*self.sb_LV - lv[dnn].Mpsc_param[dnn,i,t].value*self.sb_LV + lv[dnn].u_psc_param[bd,i,t].value*self.sb_LV)**2  for dnn in self.Boundary_Buses for (bd,i) in lv[dnn].Omega_S for t in lv[dnn].T_L)
            FO_LV_sum = FO_LV_sum_1 - Penalizater

            for dnn in mv.Omega_B:
                for (bd,i) in lv[dnn].Omega_S:
                    for t in lv[dnn].T_L:
                        mv.Lpbc_param[bd,t] = lv[dnn].Lpbc[bd,i,t].value
                        mv.db_M_param[bd,t] = lv[dnn].db[bd,i,t].value
                        mv.Lps_param[bd,t] = lv[dnn].Lps[bd,i,t].value
                        
            Mpsc_prev = { (i,t): mv.Mpsc[i,t].value for i in mv.Omega_B for t in mv.T_M }

            self.opt.solve(mv)

            FO_MV = value(mv.FunObj)
            FO_total = FO_LV_sum + FO_MV

            dict_npfe_r, dict_db_const, dict_db_const_param2, dict_pse_pg, dict_psc_pv = {}, {}, {}, {}, {} 
            
            for dnn in self.Boundary_Buses:
                for (bd,i) in lv[dnn].Omega_S:
                    for t in lv[dnn].T_L:
                        Lpbc  = lv[dnn].Lpbc[bd,i,t].value
                        Mpsc = mv.Mpsc[dnn,t].value

                        Lpbe = lv[dnn].Lpbe[bd,i,t].value
                        Lps = lv[dnn].Lps[bd,i,t].value

                        Mpse = mv.Mpse[dnn,t].value
                        Mpb  = mv.Mpb[dnn,t].value

                        lv[dnn].Lambda_L_BC_Dual[bd,i,t] = round(mv.dual[mv.npfe_r[dnn,t]],self.Dual_Decimals)
                        lv[dnn].Lambda_L_BE_Dual[bd,i,t] = round(mv.dual[mv.npfe_r[dnn,t]] - mv.dual[mv.pse_pg[t]],self.Dual_Decimals)
                        mv.Lambda_L_SC_Dual[dnn,t] = round(mv.dual[mv.npfe_r[dnn,t]],self.Dual_Decimals)
                        mv.Lambda_L_SE_Dual[dnn,t] = round(mv.dual[mv.npfe_r[dnn,t]] - mv.dual[mv.pse_pg[t]],self.Dual_Decimals)

                        dict_npfe_r[dnn,t] = round(mv.dual[mv.npfe_r[dnn,t]],self.Dual_Decimals)
                        dict_db_const[dnn,t] = round(mv.dual[mv.db_const[dnn,t]],self.Dual_Decimals)
                        dict_db_const_param2[dnn,t] = round(mv.dual[mv.db_const_param2[dnn,t]],self.Dual_Decimals)
                        dict_pse_pg[t] = round(mv.dual[mv.pse_pg[t]],self.Dual_Decimals)
                        dict_psc_pv[t] = round(mv.dual[mv.psc_pv[t]],self.Dual_Decimals)

                        u_psc_new = lv[dnn].u_psc_param[bd,i,t] + self.ADMM_u_step*(Lpbc - Mpsc)
                        Delta_pbc = Lpbc - Mpsc

                        lambda_pb_new = mv.lambda_pb_param[dnn,t] + eta*(Mpb - Lps)
                        lambda_pse_new = mv.lambda_pse_param[dnn,t] + eta*(Mpse - Lpbe)

                        lv[dnn].u_psc_param[bd,i,t] = u_psc_new
                        mv.u_psc_param[dnn,t] = u_psc_new

                        mv.lambda_pb_param[dnn,t] = lambda_pb_new
                        mv.lambda_pse_param[dnn,t] = lambda_pse_new

            res_prim, res_dual = 0, 0

            for dnn in self.Boundary_Buses:
                for (bd,i) in lv[dnn].Omega_S:
                    for t in lv[dnn].T_L:
                        Lpbc  = lv[dnn].Lpbc[bd,i,t].value

                        Mpsc = mv.Mpsc[dnn,t].value

                        res_prim += (Lpbc - Mpsc)**2 

            res_prim = np.sqrt(res_prim)

            for i in mv.Omega_B:
                for t in mv.T_M:
                    res_dual += (mv.Mpsc[i,t].value - Mpsc_prev[i,t])**2 
            res_dual = np.sqrt(res_dual) * mv.Rho


            GAP = max(res_prim, res_dual)
            print(f"Iter {k}: GAP = {GAP:.4e}, r={res_prim:.4e}, s={res_dual:.4e}, "f"FO_LVs = {FO_LV_sum:.4f}, FO_MV = {FO_MV:.4f}, Total = {FO_total:.4f}")
            if GAP < tol_gap or k >= limititer:
                break

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"\nTiempo total hasta convergencia: {total_time:.4f} segundos")

        return lv, mv, dict_npfe_r, dict_db_const, dict_db_const_param2, dict_pse_pg, dict_psc_pv

    def Print_Results_LV(self,lv,dnn):
        PV_Table_LV = pd.DataFrame()
        PL_Table_LV = pd.DataFrame()
        SOC_Table_LV = pd.DataFrame()
        CH_Table_LV = pd.DataFrame()
        DS_Table_LV = pd.DataFrame()
        W_Table_LV = pd.DataFrame()
        PSMV_Table_LV = pd.DataFrame()
        PBE_Table_LV = pd.DataFrame()
        PBC_Table_LV = pd.DataFrame()


        r=0
        for (bd,i) in lv.Omega_S:
            for t in lv.T_L:
                PSMV_Table_LV.loc[r,'Bd'] = bd
                PSMV_Table_LV.loc[r,'Bus'] = i
                PSMV_Table_LV.loc[r,t] = round(lv.Lps[bd,i,t].value,self.Result_Decimals)*lv.sb
                PBE_Table_LV.loc[r,'Bd'] = bd
                PBE_Table_LV.loc[r,'Bus'] = i
                PBE_Table_LV.loc[r,t] = round(lv.Lpbe[bd,i,t].value,self.Result_Decimals)*lv.sb

                PBC_Table_LV.loc[r,'Bd'] = bd
                PBC_Table_LV.loc[r,'Bus'] = i
                PBC_Table_LV.loc[r,t] = round(lv.Lpbc[bd,i,t].value,self.Result_Decimals)*lv.sb
            r += 1

        r=0
        for (bd,i) in lv.A:
            for t in lv.T_L:
                PV_Table_LV.loc[r,'Bd'] = bd
                PV_Table_LV.loc[r,'Bus'] = i
                PV_Table_LV.loc[r,t] = round(lv.pv[bd,i,t].value,self.Result_Decimals)*lv.sb
                PL_Table_LV.loc[r,'Bd'] = bd
                PL_Table_LV.loc[r,'Bus'] = i
                PL_Table_LV.loc[r,t] = lv.PL[bd,i,t]
                SOC_Table_LV.loc[r,'Bd'] = bd
                SOC_Table_LV.loc[r,'Bus'] = i
                SOC_Table_LV.loc[r,t] = round(lv.soc[bd,i,t].value,self.Result_Decimals)*lv.sb
                CH_Table_LV.loc[r,'Bd'] = bd
                CH_Table_LV.loc[r,'Bus'] = i
                CH_Table_LV.loc[r,t] = round(lv.ch[bd,i,t].value,self.Result_Decimals)*lv.sb
                DS_Table_LV.loc[r,'Bd'] = bd
                DS_Table_LV.loc[r,'Bus'] = i
                DS_Table_LV.loc[r,t] = round(lv.ds[bd,i,t].value,self.Result_Decimals)*lv.sb
                W_Table_LV.loc[r,'Bd'] = bd
                W_Table_LV.loc[r,'Bus'] = i
                W_Table_LV.loc[r,t] = round(lv.w[bd,i,t].value,self.Result_Decimals)
            r += 1

        with pd.ExcelWriter(self.Results_LV_File.format(dnn=dnn)) as data: 
            PSMV_Table_LV.to_excel(data, sheet_name="PSMV")
            PBE_Table_LV.to_excel(data, sheet_name="PBE")
            PBC_Table_LV.to_excel(data, sheet_name="PBC")
            PV_Table_LV.to_excel(data, sheet_name="PV")
            PL_Table_LV.to_excel(data, sheet_name="PL")
            SOC_Table_LV.to_excel(data, sheet_name="SOC")
            CH_Table_LV.to_excel(data, sheet_name="CH")
            DS_Table_LV.to_excel(data, sheet_name="DS")
            W_Table_LV.to_excel(data, sheet_name="W")

    def Print_Results_MV(self,mv,dict_npfe_r, dict_db_const, dict_db_const_param2, dict_pse_pg, dict_psc_pv):
        PG_Table_MV = pd.DataFrame()
        PV_Table_MV = pd.DataFrame()
        PL_Table_MV = pd.DataFrame()
        V_Table_MV = pd.DataFrame()
        Flow_Table_MV = pd.DataFrame()
        PBLV_Table_MV = pd.DataFrame()
        PSC_Table_MV = pd.DataFrame()
        PSE_Table_MV = pd.DataFrame()
        SOC_Table_MV = pd.DataFrame()
        CH_Table_MV = pd.DataFrame()
        DS_Table_MV = pd.DataFrame()
        Aux_load_Table_MV = pd.DataFrame()

        DUAL_db_const = pd.DataFrame()
        DUAL_pse_pg   = pd.DataFrame()
        DUAL_psc_pv   = pd.DataFrame()
        DUAL_npfe_r   = pd.DataFrame()
        DUAL_const_param2   = pd.DataFrame()
        DUAL_Final = pd.DataFrame()

        r=0
        for i in mv.Omega_B:
            for t in mv.T_M:
                PBLV_Table_MV.loc[r,'bus'] = i
                PBLV_Table_MV.loc[r,t] = round(mv.Mpb[i,t].value,self.Result_Decimals)*self.sb_MV
                PSC_Table_MV.loc[r,'bus'] = i
                PSC_Table_MV.loc[r,t] = round(mv.Mpsc[i,t].value,self.Result_Decimals)*self.sb_MV
                PSE_Table_MV.loc[r,'bus'] = i
                PSE_Table_MV.loc[r,t] = round(mv.Mpse[i,t].value,self.Result_Decimals)*self.sb_MV
            r += 1

        r=0
        for i in mv.Omega_M:
            for t in mv.T_M:
                Aux_load_Table_MV.loc[r,'bus'] = i
                Aux_load_Table_MV.loc[r,t] = round(mv.aux_load[i,t].value,self.Result_Decimals)*self.sb_MV
            r += 1

        r=0
        for i in mv.A_M:
            for t in mv.T_M:
                PG_Table_MV.loc[r,'bus'] = i
                PG_Table_MV.loc[r,t] = round(mv.pg[i,t].value,self.Result_Decimals)*self.sb_MV
                PV_Table_MV.loc[r,'bus'] = i
                PV_Table_MV.loc[r,t] = round(mv.pv_M[i,t].value,self.Result_Decimals)*self.sb_MV
                PL_Table_MV.loc[r,'bus'] = i
                PL_Table_MV.loc[r,t] = mv.PL[i,t]

            r += 1


        r=0
        for (i,j) in mv.L_M:
            for t in mv.T_M:
                Flow_Table_MV.loc[r,'from'] = str(i)
                Flow_Table_MV.loc[r,'to'] = str(j)
                Flow_Table_MV.loc[r,t] = round(mv.P[i,j,t].value,self.Flow_Result_Decimals)*self.sb_MV
            r += 1  

        r = 0
        for dnn in self.Boundary_Buses:
            for t in mv.T_M:
                # === DB_CONST ===
                DUAL_db_const.loc[r, 'DN']  = dnn
                DUAL_db_const.loc[r, t] = dict_db_const.get((dnn, t), np.nan)

                DUAL_const_param2.loc[r, 'DN']  = dnn
                DUAL_const_param2.loc[r, t] = dict_db_const_param2.get((dnn, t), np.nan)

                # === NPFE_R ===
                DUAL_npfe_r.loc[r, 'DN']  = dnn
                DUAL_npfe_r.loc[r, t] = dict_npfe_r.get((dnn, t), np.nan)

                # === PSE_PG ===
                DUAL_pse_pg.loc[r, 'DN']  = dnn
                DUAL_pse_pg.loc[r, t] = dict_pse_pg.get(t, np.nan)

                # === PSC_PV ===
                DUAL_psc_pv.loc[r, 'DN']  = dnn
                DUAL_psc_pv.loc[r, t] = dict_psc_pv.get(t, np.nan)

                DUAL_Final.loc[r, 'DN']  = dnn
                DUAL_Final.loc[r, t] = dict_npfe_r.get((dnn, t), np.nan) + dict_db_const.get((dnn, t), np.nan)

            r += 1

        with pd.ExcelWriter(self.Results_MV_File) as data:
            PBLV_Table_MV.to_excel(data, sheet_name="PBLV")
            PSE_Table_MV.to_excel(data, sheet_name="PSE") 
            PSC_Table_MV.to_excel(data, sheet_name="PSC") 
            PG_Table_MV.to_excel(data, sheet_name="PG")
            PV_Table_MV.to_excel(data, sheet_name="PV")
            PL_Table_MV.to_excel(data, sheet_name="PL")
            V_Table_MV.to_excel(data, sheet_name="V")
            Flow_Table_MV.to_excel(data, sheet_name="Flow")
            SOC_Table_MV.to_excel(data, sheet_name="SOC")
            CH_Table_MV.to_excel(data, sheet_name="CH")
            DS_Table_MV.to_excel(data, sheet_name="DS")
            Aux_load_Table_MV.to_excel(data, sheet_name="Aux_Load")
            DUAL_db_const.to_excel(data, sheet_name='DB_CONST', index=False)
            DUAL_npfe_r.to_excel(data, sheet_name='NPFE_R', index=False)
            DUAL_pse_pg.to_excel(data, sheet_name='PSE_PG', index=False)
            DUAL_psc_pv.to_excel(data, sheet_name='PSC_PV', index=False)
            DUAL_const_param2.to_excel(data, sheet_name='DB_CONST_2', index=False)
            DUAL_Final.to_excel(data, sheet_name='Dual_Final', index=False)

def solve_lv_process(payload):
    dnn, File_LV, rho, mpsc_values, u_values = payload

    worker = Binivel_MV_LV_Problem()
    worker.Rho = rho
    worker.ReadExcelFile_LV(File_LV)
    model = worker.LV_DN_Model(dnn)

    for (bd,i) in model.Omega_S:
        for t in model.T_L:
            model.Mpsc_param[bd,i,t] = mpsc_values[(bd,i,t)]
            model.u_psc_param[bd,i,t] = u_values[(bd,i,t)]

    solver = SolverFactory(worker.Solver_Name)
    solver.options['Threads'] = worker.LV_Solver_Threads
    solver.solve(model)

    solution = {}
    for var_component in model.component_objects(Var, active=True):
        solution[var_component.name] = {
            idx: value(var_component[idx]) for idx in var_component
        }

    return dnn, value(model.FunObj), solution

if __name__ == "__main__":
    runing = Binivel_MV_LV_Problem()
    runing.ReadExcelFile_LV(runing.File_LV)
    runing.ReadExcelFile_MV(runing.File_MV)
    lv, mv, dict_npfe_r, dict_db_const, dict_db_const_param2, dict_pse_pg, dict_psc_pv = runing.Algorithm()
    runing.Print_Results_MV(mv,dict_npfe_r, dict_db_const, dict_db_const_param2, dict_pse_pg, dict_psc_pv)
    for dnn in lv:
        runing.Print_Results_LV(lv[dnn],dnn)