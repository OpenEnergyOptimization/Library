import numpy as np
import pandas as pd
from pyomo.environ import * 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Select the MV test system. All case-dependent data are defined in NETWORK_CASES.
NETWORK_CASE = "IEEE33"  # Options: "IEEE33", "IEEE69"

NETWORK_CASES = {
    "IEEE33": {
        "lv_file": "Data_206_LVDN_33MV.xlsx",
        "mv_file": "Data_33_MVDN.xlsx",
        "boundary_buses": [3, 8, 19, 21, 24, 29],
        "agents_mv": [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 25, 26, 27, 28, 30, 31, 33],
        "ms_smax_factor": 2,
    },
    "IEEE69": {
        "lv_file": "Data_206_LVDN_69MV.xlsx",
        "mv_file": "Data_69_MVDN.xlsx",
        "boundary_buses": [59, 8, 11, 12, 64, 48, 50],
        "agents_mv": [2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 65, 66, 67, 68, 69],
        "ms_smax_factor": 4,
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
RHO = 18

# LV parameters
LV_AGENTS = [7, 8, 10, 11, 18, 19, 22, 25, 26, 28, 31, 32, 38, 41, 42, 45, 46, 51, 53, 57, 58, 60, 62, 74, 83, 84, 94, 95, 109, 110, 127, 130, 133, 137, 138, 142, 144, 146, 148, 151, 153, 154, 157, 158, 160, 165, 166, 169, 170, 173, 174, 177, 178, 191, 192]
PB_L = 0.0005
BESS_EFFICIENCY = 0.95
C_PV_LV = 0.0001
C_BT_LV = 0.0001
C_PV_AUX = 0.0001
C_DS_AUX = 0.0001

# MV parameters
PB_M = 0.02
MV_BESS_UNITS = 2
TOTAL_PV_MV = 1
C_PV_MV = 0.0001
LPS_MAX = 10000
MPB_BIG_M_MULTIPLIER = 10

# KKT / Big-M parameters
BIG_M_DUAL = 1e4
DUAL_INITIAL_SC = 1
DUAL_INITIAL_SE = 10

# Solver and diagnostics
SOLVER_NAME = "gurobi"
WRITE_SOLVER_RESULTS = True
SOC_GAP_EPS = 1e-12

# Output configuration
RESULTS_KKT_FILE = "KKT_Results.xlsx"
RESULTS_MV_ONLY_FILE = "KKT_Results_MV_Only.xlsx"
SOC_GAP_DETAIL_FILE = "SOC_gap_detail.xlsx"
SOC_GAP_SUMMARY_FILE = "SOC_gap_summary.xlsx"
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
        self.Ms_smax_param = self.case["ms_smax_factor"]

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
        self.PB_M = PB_M
        self.PB_L = PB_L
        self.Fi = BESS_EFFICIENCY
        self.C_PV_LV = C_PV_LV
        self.C_BT_LV = C_BT_LV
        self.C_PV_AUX = C_PV_AUX
        self.C_DS_AUX = C_DS_AUX
        self.Total_PV_MV = TOTAL_PV_MV
        self.C_PV_MV = C_PV_MV
        self.LPS_MAX = LPS_MAX
        self.MPB_bigM_multiplier = MPB_BIG_M_MULTIPLIER
        self.BigM_dual = BIG_M_DUAL
        self.Dual_initial_SC = DUAL_INITIAL_SC
        self.Dual_initial_SE = DUAL_INITIAL_SE
        self.MV_BESS_units = MV_BESS_UNITS
        self.Solver_Name = SOLVER_NAME
        self.Write_Solver_Results = WRITE_SOLVER_RESULTS
        self.SOC_gap_eps = SOC_GAP_EPS
        self.Results_KKT_File = RESULTS_KKT_FILE
        self.Results_MV_Only_File = RESULTS_MV_ONLY_FILE
        self.SOC_Gap_Detail_File = SOC_GAP_DETAIL_FILE
        self.SOC_Gap_Summary_File = SOC_GAP_SUMMARY_FILE
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

    def Single_Level_Model (self): # LV_Model
        model = AbstractModel()

        ## -- SETs -- ##
        model.Omega_M = Set(initialize=np.arange(1,self.NBus_MV+1))
        model.Omega_B = Set(initialize=self.Boundary_Buses)
        model.T_M = Set(initialize=np.arange(1,self.Time+1))
        model.A_M = Set(initialize=self.Agents_MV)
        model.RR = Set(initialize=np.arange(self.RR))
        model.Omega_L = Set(initialize=[(dn, i) for dn in self.Boundary_Buses for i in np.arange(1,self.NBus_LV+1)])
        model.Omega_S = Set(initialize=[(dn, i) for dn in self.Boundary_Buses for i in np.arange(self.Substation,self.Substation+1)])
        model.A = Set(initialize=[(dn, i) for dn in self.Boundary_Buses for i in self.Agents])
        model.T_L = Set(initialize=np.arange(1,self.Time+1))

        def Set_L_init(model):
            return ((dn, i+1,j+1) for dn in self.Boundary_Buses for i in np.arange(0,self.NBus_LV) for j in np.arange(0,self.NBus_LV) if self.Lines_LV.loc[i,j]==1 if i<j)
        model.L_L = Set(dimen=3,initialize=Set_L_init)

        def Set_L_init_M(model):
            return ((i+1,j+1) for i in np.arange(0,self.NBus_MV) for j in np.arange(0,self.NBus_MV) if self.Lines_MV.loc[i,j]==1 if i<j)
        model.L_M = Set(dimen=2,initialize=Set_L_init_M)

        def init_Others(model):
            return sorted(list(set(model.Omega_M) - set(model.Omega_B) - set(model.A_M)))
        model.Others = Set(initialize=init_Others)
            
        ## -- PARAMETERS -- ##
        model.PGmin_L = Param(model.Omega_L, initialize = 0)

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
        model.PL_L = Param(model.A,model.T_L, rule=PL_init)

        def QL_init(model,bd,i,t):
            if i in model.A:
                return self.Bus_Data_LV['QL_'+str(bd)].loc[i-1]*self.Profiles_LV['P_'+str(i)].loc[t-1]
            else:
                return 0
        model.QL_L = Param(model.Omega_L,model.T_L, rule=QL_init)

        def PVmax_init(model,bd,i,t):
            return (self.Bus_Data_LV['DG_'+str(bd)].loc[i-1]/self.sb_LV)*self.Profiles_LV['PG_max'].loc[t-1]
        model.PVmax_L = Param(model.A,model.T_L, rule=PVmax_init)

        def Irradiance_init(model,t):
            return self.Profiles_LV['PG_max'].loc[t-1]
        model.Irradiance = Param(model.T_L, rule=Irradiance_init)

        def QGmin_init(model,bd,i):
            return self.Bus_Data_LV['QGmin'].loc[i-1]
        model.QGmin_L = Param(model.Omega_L,rule=QGmin_init)

        def QGmax_init(model,bd,i):
            return self.Bus_Data_LV['QGmax'].loc[i-1]
        model.QGmax_L = Param(model.Omega_L, rule=QGmax_init)

        def PSGmax_init(model,bd,i):
            return self.Bus_Data_LV['P_SG_max'].loc[i-1]
        model.PSG_max_L = Param(model.Omega_S,rule=PSGmax_init)

        def PBGmax_init(model,bd,i):
            return self.Bus_Data_LV['P_BG_max'].loc[i-1]
        model.PBG_max_L = Param(model.Omega_S,rule=PBGmax_init)

        def Vmax_init(model,bd,i):
            return self.Bus_Data_LV['Vmax'].loc[i-1]
        model.Vmax_L = Param(model.Omega_L, rule=Vmax_init)

        def Vmin_init(model,bd,i):
            return self.Bus_Data_LV['Vmin'].loc[i-1]
        model.Vmin_L = Param(model.Omega_L, rule=Vmin_init)

        def Smax_init(model,bd,i,j):
            return self.Smax_LV.loc[i-1,j-1]
        model.s_max_L = Param(model.L_L,rule=Smax_init)

        def X_init(model,bd,i,j):
            return self.X_Matrix_LV.loc[i-1,j-1]
        model.X_L = Param(model.L_L,rule=X_init)

        def R_init(model,bd,i,j):
            return self.R_Matrix_LV.loc[i-1,j-1]
        model.R_L = Param(model.L_L,rule=R_init)


        model.PB_M = self.PB_M 
        model.Cpv = self.C_PV_AUX
        model.Cds = self.C_DS_AUX

        model.SOC_min_L = self.SOC_min
        model.SOC_max_L = self.SOC_max
        model.PB_L = self.PB_L
        model.Fi = self.Fi
        model.C_PV = self.C_PV_LV
        model.C_BT = self.C_BT_LV
        model.Rho = self.Rho
        model.Total_PV_MV = self.Total_PV_MV

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

        #Parametros y variables del MV

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
        model.PL_M = Param(model.A_M,model.T_M, rule=PL_init)

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

        model.db_M_param = Var(model.Omega_B, model.T_M, within=Reals, bounds=(self.LB, self.UB))
        model.Lps_param  = Var(model.Omega_B, model.T_M, within=NonNegativeReals, bounds=(0, self.UB))

        model.sb = Param(initialize = self.sb_MV)
        model.SOC_min_M = self.SOC_min
        model.SOC_max_M = self.SOC_max
        model.C_pv = self.C_PV_MV

        ## -- VARIABLES -- ##
        model.pg_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.pv_M = Var(model.A_M,model.T_M, within = NonNegativeReals, initialize = 0)
        model.qg_M = Var(model.Omega_M,model.T_M,initialize = 0)
        model.V_M = Var(model.Omega_M,model.T_M, initialize = 0)
        model.P_M = Var(model.L_M,model.T_M, initialize = 1)
        model.Q_M = Var(model.L_M,model.T_M, initialize = 0)
        model.l_M = Var(model.L_M,model.T_M)

        model.Mpb = Var(model.Omega_B,model.T_M, within = NonNegativeReals, initialize = 0)
        model.Mpsc = Var(model.Omega_B,model.T_M, within = NonNegativeReals, initialize = 0)
        model.Mpse = Var(model.Omega_B,model.T_M, within = NonNegativeReals, initialize = 0)

        model.db_M = Var(model.Omega_B,model.T_M, within = Reals, initialize = 0)
        model.dp_M = Var(model.A_M,model.T_M, within = Reals, initialize = 0)

        model.LPS_MAX = Param(initialize=self.LPS_MAX)

        ## -- OBJECTIVE FUNCTION -- ##
        def Fun_obj(model):
            return sum((model.Lambda_L_BC[t])*model.Mpsc[bd,t]*model.sb + (model.Lambda_L_BE[t])*model.Mpse[bd,t]*model.sb  for bd in model.Omega_B for t in model.T_L) - \
                   sum(model.Lambda_L_S[t]*model.Lps[bd,i,t]*model.sb for (bd,i) in model.Omega_S for t in model.T_L) + \
                   sum(model.C_PV*model.pv[bd,i,t]*model.sb + model.C_BT*model.ds[bd,i,t]*model.sb for (bd,i) in model.A for t in model.T_L) 
        model.FunObj = Objective(rule = Fun_obj, sense = minimize)

        # -- CONSTRAINTS -- ##

        def NPFE_R (model,bd,i,t):
            value1 = sum(model.P[bdd,m,n,t] for (bdd,m,n) in model.L_L if m==i if bdd==bd) - sum(model.P[bdd,n,m,t] - model.R_L[bdd,n,m]*model.l[bdd,n,m,t] for (bdd,n,m) in model.L_L if m==i if bdd==bd) 
            if (bd,i) in model.A:
                return model.dp[bd,i,t] == value1 
            if (bd,i) in model.Omega_S:
                return model.db[bd,i,t]   == value1 
            else:
                return 0 == value1
        model.npfe_r = Constraint(model.Omega_L,model.T_L, rule=NPFE_R)

        def NPFE_Im (model,bd,i,t):
            value2 = sum(model.Q[bdd,m,n,t] for (bdd,m,n) in model.L_L if m==i if bdd==bd) - sum(model.Q[bdd,n,m,t] - model.X_L[bdd,n,m]*model.l[bdd,n,m,t] for (bdd,n,m) in model.L_L if m==i if bdd==bd) 
            return model.qg[bd,i,t] - model.QL_L[bd,i,t]/self.sb_LV == value2
        model.npfe_im = Constraint(model.Omega_L,model.T_L, rule=NPFE_Im)

        def Market_Constraint_0 (model,bd,i,t): 
            return model.dp[bd,i,t] == model.pv[bd,i,t] - model.PL_L[bd,i,t]/self.sb_LV + model.ds[bd,i,t] - model.ch[bd,i,t] 
        model.market_constraint_0 = Constraint(model.A,model.T_L, rule= Market_Constraint_0)

        def Market_Constraint_1 (model,bd,i,t): 
            return model.db[bd,i,t] == model.Mpsc[bd,t] + model.Mpse[bd,t] - model.Lps[bd,i,t]
        model.market_constraint_1 = Constraint(model.Omega_S,model.T_L, rule= Market_Constraint_1)

        def db_M_Const_link_1(model, bd, t):
            return model.db_M_param[bd,t] == sum(model.db[bd2,i,t]  for (bd2,i) in model.Omega_S if bd2 == bd) * self.sb_LV
        model.db_M_const_link_1 = Constraint(model.Omega_B, model.T_M, rule=db_M_Const_link_1)

        def Market_Constraint_4 (model,bd,i,t): 
            return model.Mpsc[bd,t]  <= (model.Total_PV_MV/self.sb_LV*model.Irradiance[t] + self.MV_BESS_units*self.PB_M/self.sb_LV)
        model.market_constraint_4 = Constraint(model.Omega_S,model.T_L, rule= Market_Constraint_4)

        def LPFE_R (model, bd,i,j,t):
            return model.V[bd,j,t] == model.V[bd,i,t] - 2*(model.R_L[bd,i,j]*model.P[bd,i,j,t] + model.X_L[bd,i,j]*model.Q[bd,i,j,t]) + (model.R_L[bd,i,j]**2 + model.X_L[bd,i,j]**2)*model.l[bd,i,j,t]
        model.lpfe_r = Constraint(model.L_L,model.T_L, rule=LPFE_R)

        def LPFE_Iq (model, bd,i,j,t):
            return (model.P[bd,i,j,t]**2 + model.Q[bd,i,j,t]**2) <= model.l[bd,i,j,t]*model.V[bd,i,t]
        model.lpfe_iq = Constraint(model.L_L,model.T_L, rule=LPFE_Iq)

        def GL_QG (model, bd,i,t):
            return inequality(model.QGmin_L[bd,i]/self.sb_LV, model.qg[bd,i,t],model.QGmax_L[bd,i]/self.sb_LV)
        model.gl_qg = Constraint(model.Omega_L,model.T_L, rule=GL_QG)

        def VL (model,bd,i,t):
            return inequality(model.Vmin_L[bd,i]**2, model.V[bd,i,t], model.Vmax_L[bd,i]**2)
        model.vl = Constraint(model.Omega_L,model.T_L, rule=VL)

        def GL_PG (model, bd,i,t):
            return model.pv[bd,i,t] <= model.PVmax_L[bd,i,t]
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


        ##########################
        ## -- CONSTRAINTS  MV-- ##
        ##########################

        def NPFE_R (model,i,t):
            value1 = sum(model.P_M[m,n,t] for (m,n) in model.L_M if m==i) - sum(model.P_M[n,m,t] for (n,m) in model.L_M if m==i) 
            if i in model.Omega_B:
                return model.db_M[i,t]  == value1 
            if i in model.A_M:
                return model.dp_M[i,t]  == value1 
            else:
                return 0 == value1
        model.npfe_r_M = Constraint(model.Omega_M,model.T_M, rule=NPFE_R)

        def NPFE_Im (model,i,t):
            value2 = sum(model.Q_M[m,n,t] for (m,n) in model.L_M if m==i) - sum(model.Q_M[n,m,t] for (n,m) in model.L_M if m==i) 
            if i in model.A_M:
                return model.qg_M[i,t] - model.QL[i,t]/self.sb_MV == value2
            else:
                return model.qg_M[i,t] == value2
        model.npfe_im_M = Constraint(model.Omega_M,model.T_M, rule=NPFE_Im)

        def DP_Const (model, i,t):
            return model.dp_M[i,t] - model.pg_M[i,t] - model.pv_M[i,t] == - model.PL_M[i,t]/self.sb_MV 
        model.dp_const_M = Constraint(model.A_M,model.T_M, rule=DP_Const)

        def DB_Const (model, i,t):
            return model.db_M[i,t] - model.Mpb[i,t] + model.Mpsc[i,t] + model.Mpse[i,t] == 0
        model.db_const_M = Constraint(model.Omega_B,model.T_M, rule=DB_Const)

        def DB_Const_Param (model, i,t):
            return model.db_M[i,t] == -model.db_M_param[i,t]/self.sb_MV 
        model.db_const_param_M = Constraint(model.Omega_B,model.T_M, rule=DB_Const_Param)

        def DB_Const_Param2 (model, i,t):
            return model.Mpb[i,t] <= model.Lps_param[i,t]/self.sb_MV 
        model.db_const_param2_M = Constraint(model.Omega_B,model.T_M, rule=DB_Const_Param2)

        def LPFE_R (model, i,j,t):
            return model.V_M[j,t] - model.V_M[i,t] + 2*(model.R[i,j]*model.P_M[i,j,t] + model.X[i,j]*model.Q_M[i,j,t]) == 0
        model.lpfe_r_M = Constraint(model.L_M,model.T_M, rule=LPFE_R)

        def GL_PG (model, i,t):
            return model.pg_M[i,t] <= model.PGmax[i,t]/self.sb_MV
        model.gl_pg_M = Constraint(model.A_M,model.T_M, rule=GL_PG)

        def GL_PV (model, i,t):
            return model.pv_M[i,t] <= model.PVmax[i,t]/self.sb_MV
        model.gl_pv_M = Constraint(model.A_M,model.T_M, rule=GL_PV)

        def GL_QG1 (model, i,t):
            return model.qg_M[i,t] >= model.QGmin[i]/self.sb_MV
        model.gl_qg1_M = Constraint(model.Omega_M,model.T_M, rule=GL_QG1)

        def GL_QG2 (model, i,t):
            return model.qg_M[i,t] <= model.QGmax[i]/self.sb_MV
        model.gl_qg2_M = Constraint(model.Omega_M,model.T_M, rule=GL_QG2)

        def VL1 (model,i,t):
            return model.V_M[i,t] >= model.Vmin[i]**2
        model.vl1_M = Constraint(model.Omega_M,model.T_M, rule=VL1)

        def VL2 (model,i,t):
            return model.V_M[i,t] <= model.Vmax[i]**2
        model.vl2_M = Constraint(model.Omega_M,model.T_M, rule=VL2)

        def PSE_PG(model,t):
            return sum(model.Mpse[i,t] for i in model.Omega_B) <= sum(model.pg_M[i,t] for i in model.A_M)
        model.pse_pg_M = Constraint(model.T_M, rule = PSE_PG)

        def PSC_PV(model,t):
            return sum(model.Mpsc[i,t] for i in model.Omega_B) <= sum(model.pv_M[i,t]  for i in model.A_M) 
        model.psc_pv_M = Constraint(model.T_M, rule = PSC_PV)

        def Smax_Limit (model,i,j,t,r):
            return  model.P_M[i,j,t]*model.AA[r] + model.Q_M[i,j,t]*model.BB[r] + (model.s_max[i,j]/self.sb_MV)*model.CC[r] <= 0
        model.smax_limit = Constraint(model.L_M,model.T_M, model.RR, rule=Smax_Limit)

        # ============================================================
        #   KKT - (1) DUALES NECESARIOS PARA STATIONARITY
        # ============================================================

        # ---- Igualdades (multiplicadores libres) ----
        model.lam_npfe_r   = Var(model.Omega_M, model.T_M, within=Reals)   # npfe_r
        model.lam_npfe_im  = Var(model.Omega_M, model.T_M, within=Reals)   # npfe_im
        model.lam_dp_const = Var(model.A_M,     model.T_M, within=Reals)   # dp_const
        model.lam_db_const = Var(model.Omega_B, model.T_M, within=Reals)   # db_const
        model.lam_db_par   = Var(model.Omega_B, model.T_M, within=Reals)   # db_const_param
        model.lam_lpfe_r   = Var(model.L_M,     model.T_M, within=Reals)   # lpfe_r

        model.mu_pg_max = Var(model.A_M, model.T_M, within=NonNegativeReals)
        model.mu_pv_max = Var(model.A_M, model.T_M, within=NonNegativeReals)
        model.mu_qg_min = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.mu_qg_max = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.mu_V_min  = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.mu_V_max  = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.mu_Mpb_max = Var(model.Omega_B, model.T_M, within=NonNegativeReals)
        model.mu_pse = Var(model.T_M, within=NonNegativeReals)
        model.mu_psc = Var(model.T_M, within=NonNegativeReals)

        model.mu_pg_min   = Var(model.A_M,     model.T_M, within=NonNegativeReals)  # pg >= 0
        model.mu_pv_min   = Var(model.A_M,     model.T_M, within=NonNegativeReals)  # pv >= 0
        model.mu_Mpb_min  = Var(model.Omega_B, model.T_M, within=NonNegativeReals)  # Mpb >= 0
        model.mu_Mpsc_min = Var(model.Omega_B, model.T_M, within=NonNegativeReals)  # Mpsc >= 0
        model.mu_Mpse_min = Var(model.Omega_B, model.T_M, within=NonNegativeReals)  # Mpse >= 0

        model.mu_smax = Var(model.L_M,model.T_M,model.RR,within=NonNegativeReals)


        # ============================================================
        #   KKT - STATIONARITY
        # ============================================================
        def ST_pg(m, i, t):
            return (2*m.Ca[i]*(self.sb_MV**2)*m.pg_M[i,t] + m.Cb[i]*self.sb_MV
                    - m.lam_dp_const[i,t]
                    + m.mu_pg_max[i,t] - m.mu_pg_min[i,t]
                    - m.mu_pse[t]) == 0
        model.st_pg = Constraint(model.A_M, model.T_M, rule=ST_pg)

        def ST_pv(m, i, t):
            return (self.sb_MV*m.C_pv
                    - m.lam_dp_const[i,t]
                    + m.mu_pv_max[i,t] - m.mu_pv_min[i,t]
                    - m.mu_psc[t]) == 0
        model.st_pv = Constraint(model.A_M, model.T_M, rule=ST_pv)

        def ST_dpM(m, i, t):
            return (m.lam_npfe_r[i,t] + m.lam_dp_const[i,t]) == 0
        model.st_dpM = Constraint(model.A_M, model.T_M, rule=ST_dpM)

        def ST_dbM(m, b, t):
            return (m.lam_npfe_r[b,t] + m.lam_db_const[b,t] + m.lam_db_par[b,t]) == 0
        model.st_dbM = Constraint(model.Omega_B, model.T_M, rule=ST_dbM)

        def ST_Mpb(m, b, t):
            return (-m.lam_db_const[b,t]
                    + m.mu_Mpb_max[b,t] - m.mu_Mpb_min[b,t]) == 0
        model.st_Mpb = Constraint(model.Omega_B, model.T_M, rule=ST_Mpb)

        def ST_Mpsc(m, b, t):
            return (m.lam_db_const[b,t]
                    + m.mu_psc[t]
                    - m.mu_Mpsc_min[b,t]) == 0
        model.st_Mpsc = Constraint(model.Omega_B, model.T_M, rule=ST_Mpsc)

        def ST_Mpse(m, b, t):
            return (m.lam_db_const[b,t]
                    + m.mu_pse[t]
                    - m.mu_Mpse_min[b,t]) == 0
        model.st_Mpse = Constraint(model.Omega_B, model.T_M, rule=ST_Mpse)

        def ST_qg(m, i, t):
            return (m.lam_npfe_im[i,t]
                    - m.mu_qg_min[i,t] + m.mu_qg_max[i,t]) == 0
        model.st_qg = Constraint(model.Omega_M, model.T_M, rule=ST_qg)

        def ST_V(m, i, t):
            out_terms = sum(m.lam_lpfe_r[i,j,t] for (i2,j) in m.L_M if i2 == i)
            in_terms  = sum(m.lam_lpfe_r[j,i,t] for (j,i2) in m.L_M if i2 == i)
            return (in_terms - out_terms
                    - m.mu_V_min[i,t] + m.mu_V_max[i,t]) == 0
        model.st_V = Constraint(model.Omega_M, model.T_M, rule=ST_V)

        def ST_P(m, i, j, t):
            smax_term = sum(m.AA[r] * m.mu_smax[i,j,t,r]for r in m.RR)
            return (-m.lam_npfe_r[i,t] + m.lam_npfe_r[j,t]
                    + 2*m.R[i,j]*m.lam_lpfe_r[i,j,t]) + smax_term == 0
        model.st_P = Constraint(model.L_M, model.T_M, rule=ST_P)

        def ST_Q(m, i, j, t):
            smax_term = sum(m.BB[r] * m.mu_smax[i,j,t,r]for r in m.RR)
            return (-m.lam_npfe_im[i,t] + m.lam_npfe_im[j,t]
                    + 2*m.X[i,j]*m.lam_lpfe_r[i,j,t]) + smax_term == 0
        model.st_Q = Constraint(model.L_M, model.T_M, rule=ST_Q)


        # ============================================================
        #   KKT - (3) SLACKNESS 
        # ============================================================

        # --------------------------
        # Big-M para DUALES 
        # --------------------------
        model.Mmu_pg   = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_pv   = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_qg   = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_V    = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_Mpb  = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_pse  = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_psc  = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_pgmin   = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_pvmin   = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_Mpbmin  = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_Mpscmin = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_Mpsemin = Param(initialize=self.BigM_dual, mutable=True)
        model.Mmu_smax = Param(initialize=self.BigM_dual, mutable=True)

        def Ms_smax_init(m, i, j, t, r):
            return self.Ms_smax_param*(m.s_max[i,j]/self.sb_MV)
        model.Ms_smax = Param(model.L_M,model.T_M,model.RR,initialize=Ms_smax_init)

        def Ms_pg_max_init(m, i, t):
            return m.PGmax[i, t] / self.sb_MV
        model.Ms_pg_max = Param(model.A_M, model.T_M, initialize=Ms_pg_max_init)

        def Ms_pv_max_init(m, i, t):
            return m.PVmax[i, t] / self.sb_MV
        model.Ms_pv_max = Param(model.A_M, model.T_M, initialize=Ms_pv_max_init)

        def Ms_qg_rng_init(m, i, t):
            return (m.QGmax[i] - m.QGmin[i]) / self.sb_MV
        model.Ms_qg_rng = Param(model.Omega_M, model.T_M, initialize=Ms_qg_rng_init)

        def Ms_V_rng_init(m, i, t):
            return (m.Vmax[i]**2 - m.Vmin[i]**2)
        model.Ms_V_rng = Param(model.Omega_M, model.T_M, initialize=Ms_V_rng_init)

        def Ms_Mpb_max_init(m, b, t):
            return m.LPS_MAX*self.MPB_bigM_multiplier / self.sb_MV
        model.Ms_Mpb_max = Param(model.Omega_B, model.T_M, initialize=Ms_Mpb_max_init)

        def TotPGmax_init(m, t):
            return sum(m.PGmax[i, t] / self.sb_MV for i in m.A_M)
        model.TotPGmax = Param(model.T_M, initialize=TotPGmax_init)

        def TotPVmax_init(m, t):
            return sum(m.PVmax[i, t] / self.sb_MV for i in m.A_M)
        model.TotPVmax = Param(model.T_M, initialize=TotPVmax_init)

        model.Ms_pse = Param(model.T_M, initialize=lambda m,t: m.TotPGmax[t])
        model.Ms_psc = Param(model.T_M, initialize=lambda m,t: m.TotPVmax[t])

        model.Ms_pg_min = Param(model.A_M, model.T_M, initialize=Ms_pg_max_init)
        model.Ms_pv_min = Param(model.A_M, model.T_M, initialize=Ms_pv_max_init)
        model.Ms_Mpb_min = Param(model.Omega_B, model.T_M, initialize=Ms_Mpb_max_init)
        model.Ms_Mpsc_min = Param(model.Omega_B, model.T_M, initialize=lambda m,b,t: m.TotPVmax[t])
        model.Ms_Mpse_min = Param(model.Omega_B, model.T_M, initialize=lambda m,b,t: m.TotPGmax[t])

        # ------------------------------------------------------------
        # Complementariedad
        # ------------------------------------------------------------

        model.s_pg_max = Var(model.A_M, model.T_M, within=NonNegativeReals)
        model.z_pg_max = Var(model.A_M, model.T_M, within=Binary)

        def SL_pg_max(m, i, t):
            return (m.pg_M[i,t] - m.PGmax[i,t]/self.sb_MV) + m.s_pg_max[i,t] == 0
        model.sl_pg_max = Constraint(model.A_M, model.T_M, rule=SL_pg_max)

        def FA_pg_max_mu(m, i, t):
            return m.mu_pg_max[i,t] <= m.Mmu_pg * m.z_pg_max[i,t]
        def FA_pg_max_s(m, i, t):
            return m.s_pg_max[i,t] <= m.Ms_pg_max[i,t] * (1 - m.z_pg_max[i,t])
        model.fa_pg_max_mu = Constraint(model.A_M, model.T_M, rule=FA_pg_max_mu)
        model.fa_pg_max_s  = Constraint(model.A_M, model.T_M, rule=FA_pg_max_s)

        model.s_pv_max = Var(model.A_M, model.T_M, within=NonNegativeReals)
        model.z_pv_max = Var(model.A_M, model.T_M, within=Binary)

        def SL_pv_max(m, i, t):
            return (m.pv_M[i,t] - m.PVmax[i,t]/self.sb_MV) + m.s_pv_max[i,t] == 0
        model.sl_pv_max = Constraint(model.A_M, model.T_M, rule=SL_pv_max)

        def FA_pv_max_mu(m, i, t):
            return m.mu_pv_max[i,t] <= m.Mmu_pv * m.z_pv_max[i,t]
        def FA_pv_max_s(m, i, t):
            return m.s_pv_max[i,t] <= m.Ms_pv_max[i,t] * (1 - m.z_pv_max[i,t])
        model.fa_pv_max_mu = Constraint(model.A_M, model.T_M, rule=FA_pv_max_mu)
        model.fa_pv_max_s  = Constraint(model.A_M, model.T_M, rule=FA_pv_max_s)

        model.s_qg_min = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.z_qg_min = Var(model.Omega_M, model.T_M, within=Binary)

        def SL_qg_min(m, i, t):
            return (m.QGmin[i]/self.sb_MV - m.qg_M[i,t]) + m.s_qg_min[i,t] == 0
        model.sl_qg_min = Constraint(model.Omega_M, model.T_M, rule=SL_qg_min)

        def FA_qg_min_mu(m, i, t):
            return m.mu_qg_min[i,t] <= m.Mmu_qg * m.z_qg_min[i,t]
        def FA_qg_min_s(m, i, t):
            return m.s_qg_min[i,t] <= m.Ms_qg_rng[i,t] * (1 - m.z_qg_min[i,t])
        model.fa_qg_min_mu = Constraint(model.Omega_M, model.T_M, rule=FA_qg_min_mu)
        model.fa_qg_min_s  = Constraint(model.Omega_M, model.T_M, rule=FA_qg_min_s)

        model.s_qg_max = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.z_qg_max = Var(model.Omega_M, model.T_M, within=Binary)

        def SL_qg_max(m, i, t):
            return (m.qg_M[i,t] - m.QGmax[i]/self.sb_MV) + m.s_qg_max[i,t] == 0
        model.sl_qg_max = Constraint(model.Omega_M, model.T_M, rule=SL_qg_max)

        def FA_qg_max_mu(m, i, t):
            return m.mu_qg_max[i,t] <= m.Mmu_qg * m.z_qg_max[i,t]
        def FA_qg_max_s(m, i, t):
            return m.s_qg_max[i,t] <= m.Ms_qg_rng[i,t] * (1 - m.z_qg_max[i,t])
        model.fa_qg_max_mu = Constraint(model.Omega_M, model.T_M, rule=FA_qg_max_mu)
        model.fa_qg_max_s  = Constraint(model.Omega_M, model.T_M, rule=FA_qg_max_s)

        model.s_V_min = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.z_V_min = Var(model.Omega_M, model.T_M, within=Binary)

        def SL_V_min(m, i, t):
            return (m.Vmin[i]**2 - m.V_M[i,t]) + m.s_V_min[i,t] == 0
        model.sl_V_min = Constraint(model.Omega_M, model.T_M, rule=SL_V_min)

        def FA_V_min_mu(m, i, t):
            return m.mu_V_min[i,t] <= m.Mmu_V * m.z_V_min[i,t]
        def FA_V_min_s(m, i, t):
            return m.s_V_min[i,t] <= m.Ms_V_rng[i,t] * (1 - m.z_V_min[i,t])
        model.fa_V_min_mu = Constraint(model.Omega_M, model.T_M, rule=FA_V_min_mu)
        model.fa_V_min_s  = Constraint(model.Omega_M, model.T_M, rule=FA_V_min_s)

        model.s_V_max = Var(model.Omega_M, model.T_M, within=NonNegativeReals)
        model.z_V_max = Var(model.Omega_M, model.T_M, within=Binary)

        def SL_V_max(m, i, t):
            return (m.V_M[i,t] - m.Vmax[i]**2) + m.s_V_max[i,t] == 0
        model.sl_V_max = Constraint(model.Omega_M, model.T_M, rule=SL_V_max)

        def FA_V_max_mu(m, i, t):
            return m.mu_V_max[i,t] <= m.Mmu_V * m.z_V_max[i,t]
        def FA_V_max_s(m, i, t):
            return m.s_V_max[i,t] <= m.Ms_V_rng[i,t] * (1 - m.z_V_max[i,t])
        model.fa_V_max_mu = Constraint(model.Omega_M, model.T_M, rule=FA_V_max_mu)
        model.fa_V_max_s  = Constraint(model.Omega_M, model.T_M, rule=FA_V_max_s)

        model.s_Mpb_max = Var(model.Omega_B, model.T_M, within=NonNegativeReals)
        model.z_Mpb_max = Var(model.Omega_B, model.T_M, within=Binary)

        def SL_Mpb_max(m, b, t):
            return (m.Mpb[b,t] - m.Lps_param[b,t]/self.sb_MV) + m.s_Mpb_max[b,t] == 0
        model.sl_Mpb_max = Constraint(model.Omega_B, model.T_M, rule=SL_Mpb_max)

        def FA_Mpb_max_mu(m, b, t):
            return m.mu_Mpb_max[b,t] <= m.Mmu_Mpb * m.z_Mpb_max[b,t]
        def FA_Mpb_max_s(m, b, t):
            return m.s_Mpb_max[b,t] <= m.Ms_Mpb_max[b,t] * (1 - m.z_Mpb_max[b,t])
        model.fa_Mpb_max_mu = Constraint(model.Omega_B, model.T_M, rule=FA_Mpb_max_mu)
        model.fa_Mpb_max_s  = Constraint(model.Omega_B, model.T_M, rule=FA_Mpb_max_s)

        model.s_pse = Var(model.T_M, within=NonNegativeReals)
        model.z_pse = Var(model.T_M, within=Binary)

        def SL_pse(m, t):
            return (sum(m.Mpse[b,t] for b in m.Omega_B) - sum(m.pg_M[i,t] for i in m.A_M)) + m.s_pse[t] == 0
        model.sl_pse = Constraint(model.T_M, rule=SL_pse)

        def FA_pse_mu(m, t):
            return m.mu_pse[t] <= m.Mmu_pse * m.z_pse[t]
        def FA_pse_s(m, t):
            return m.s_pse[t] <= m.Ms_pse[t] * (1 - m.z_pse[t])
        model.fa_pse_mu = Constraint(model.T_M, rule=FA_pse_mu)
        model.fa_pse_s  = Constraint(model.T_M, rule=FA_pse_s)

        model.s_psc = Var(model.T_M, within=NonNegativeReals)
        model.z_psc = Var(model.T_M, within=Binary)

        def SL_psc(m, t):
            return (sum(m.Mpsc[b,t] for b in m.Omega_B) - sum(m.pv_M[i,t] for i in m.A_M)) + m.s_psc[t] == 0
        model.sl_psc = Constraint(model.T_M, rule=SL_psc)

        def FA_psc_mu(m, t):
            return m.mu_psc[t] <= m.Mmu_psc * m.z_psc[t]
        def FA_psc_s(m, t):
            return m.s_psc[t] <= m.Ms_psc[t] * (1 - m.z_psc[t])
        model.fa_psc_mu = Constraint(model.T_M, rule=FA_psc_mu)
        model.fa_psc_s  = Constraint(model.T_M, rule=FA_psc_s)

        model.s_smax = Var(model.L_M,model.T_M,model.RR,within=NonNegativeReals)
        model.z_smax = Var(model.L_M,model.T_M,model.RR,within=Binary)

        def SL_smax(m, i, j, t, r):
            return (m.P_M[i,j,t]*m.AA[r]+ m.Q_M[i,j,t]*m.BB[r]+ (m.s_max[i,j]/self.sb_MV)*m.CC[r]+ m.s_smax[i,j,t,r]== 0)
        model.sl_smax = Constraint(model.L_M,model.T_M,model.RR,rule=SL_smax)

        def FA_smax_mu(m, i, j, t, r):
            return (m.mu_smax[i,j,t,r]<= m.Mmu_smax * m.z_smax[i,j,t,r])

        def FA_smax_s(m, i, j, t, r):
            return (m.s_smax[i,j,t,r]<= m.Ms_smax[i,j,t,r]* (1 - m.z_smax[i,j,t,r]))

        model.fa_smax_mu = Constraint(model.L_M,model.T_M,model.RR,rule=FA_smax_mu)
        model.fa_smax_s = Constraint(model.L_M,model.T_M,model.RR,rule=FA_smax_s)


        # ============================================================
        #   NO NEGATIVIDAD
        # ============================================================

        model.s_pg_min = Var(model.A_M, model.T_M, within=NonNegativeReals)
        model.z_pg_min = Var(model.A_M, model.T_M, within=Binary)

        def SL_pg_min(m, i, t):
            return (-m.pg_M[i,t]) + m.s_pg_min[i,t] == 0
        model.sl_pg_min = Constraint(model.A_M, model.T_M, rule=SL_pg_min)

        def FA_pg_min_mu(m, i, t):
            return m.mu_pg_min[i,t] <= m.Mmu_pgmin * m.z_pg_min[i,t]
        def FA_pg_min_s(m, i, t):
            return m.s_pg_min[i,t] <= m.Ms_pg_min[i,t] * (1 - m.z_pg_min[i,t])
        model.fa_pg_min_mu = Constraint(model.A_M, model.T_M, rule=FA_pg_min_mu)
        model.fa_pg_min_s  = Constraint(model.A_M, model.T_M, rule=FA_pg_min_s)

        model.s_pv_min = Var(model.A_M, model.T_M, within=NonNegativeReals)
        model.z_pv_min = Var(model.A_M, model.T_M, within=Binary)

        def SL_pv_min(m, i, t):
            return (-m.pv_M[i,t]) + m.s_pv_min[i,t] == 0
        model.sl_pv_min = Constraint(model.A_M, model.T_M, rule=SL_pv_min)

        def FA_pv_min_mu(m, i, t):
            return m.mu_pv_min[i,t] <= m.Mmu_pvmin * m.z_pv_min[i,t]
        def FA_pv_min_s(m, i, t):
            return m.s_pv_min[i,t] <= m.Ms_pv_min[i,t] * (1 - m.z_pv_min[i,t])
        model.fa_pv_min_mu = Constraint(model.A_M, model.T_M, rule=FA_pv_min_mu)
        model.fa_pv_min_s  = Constraint(model.A_M, model.T_M, rule=FA_pv_min_s)

        model.s_Mpb_min = Var(model.Omega_B, model.T_M, within=NonNegativeReals)
        model.z_Mpb_min = Var(model.Omega_B, model.T_M, within=Binary)

        def SL_Mpb_min(m, b, t):
            return (-m.Mpb[b,t]) + m.s_Mpb_min[b,t] == 0
        model.sl_Mpb_min = Constraint(model.Omega_B, model.T_M, rule=SL_Mpb_min)

        def FA_Mpb_min_mu(m, b, t):
            return m.mu_Mpb_min[b,t] <= m.Mmu_Mpbmin * m.z_Mpb_min[b,t]
        def FA_Mpb_min_s(m, b, t):
            return m.s_Mpb_min[b,t] <= m.Ms_Mpb_min[b,t] * (1 - m.z_Mpb_min[b,t])
        model.fa_Mpb_min_mu = Constraint(model.Omega_B, model.T_M, rule=FA_Mpb_min_mu)
        model.fa_Mpb_min_s  = Constraint(model.Omega_B, model.T_M, rule=FA_Mpb_min_s)

        model.s_Mpsc_min = Var(model.Omega_B, model.T_M, within=NonNegativeReals)
        model.z_Mpsc_min = Var(model.Omega_B, model.T_M, within=Binary)

        def SL_Mpsc_min(m, b, t):
            return (-m.Mpsc[b,t]) + m.s_Mpsc_min[b,t] == 0
        model.sl_Mpsc_min = Constraint(model.Omega_B, model.T_M, rule=SL_Mpsc_min)

        def FA_Mpsc_min_mu(m, b, t):
            return m.mu_Mpsc_min[b,t] <= m.Mmu_Mpscmin * m.z_Mpsc_min[b,t]
        def FA_Mpsc_min_s(m, b, t):
            return m.s_Mpsc_min[b,t] <= m.Ms_Mpsc_min[b,t] * (1 - m.z_Mpsc_min[b,t])
        model.fa_Mpsc_min_mu = Constraint(model.Omega_B, model.T_M, rule=FA_Mpsc_min_mu)
        model.fa_Mpsc_min_s  = Constraint(model.Omega_B, model.T_M, rule=FA_Mpsc_min_s)

        model.s_Mpse_min = Var(model.Omega_B, model.T_M, within=NonNegativeReals)
        model.z_Mpse_min = Var(model.Omega_B, model.T_M, within=Binary)

        def SL_Mpse_min(m, b, t):
            return (-m.Mpse[b,t]) + m.s_Mpse_min[b,t] == 0
        model.sl_Mpse_min = Constraint(model.Omega_B, model.T_M, rule=SL_Mpse_min)

        def FA_Mpse_min_mu(m, b, t):
            return m.mu_Mpse_min[b,t] <= m.Mmu_Mpsemin * m.z_Mpse_min[b,t]
        def FA_Mpse_min_s(m, b, t):
            return m.s_Mpse_min[b,t] <= m.Ms_Mpse_min[b,t] * (1 - m.z_Mpse_min[b,t])
        model.fa_Mpse_min_mu = Constraint(model.Omega_B, model.T_M, rule=FA_Mpse_min_mu)
        model.fa_Mpse_min_s  = Constraint(model.Omega_B, model.T_M, rule=FA_Mpse_min_s)

        return model.create_instance()
    
    def Algorithm(self):
        sm = self.Single_Level_Model()
        self.opt = SolverFactory(self.Solver_Name)
        results = self.opt.solve(sm)
        if self.Write_Solver_Results:
            results.write()
        return sm

    def Print_Results_KKT(self,mv):
        PV_Table_LV = pd.DataFrame()
        PL_Table_LV = pd.DataFrame()
        SOC_Table_LV = pd.DataFrame()
        CH_Table_LV = pd.DataFrame()
        DS_Table_LV = pd.DataFrame()
        W_Table_LV = pd.DataFrame()
        PSMV_Table_LV = pd.DataFrame()
        PBE_Table_LV = pd.DataFrame()
        PBC_Table_LV = pd.DataFrame()

        PG_Table_MV = pd.DataFrame()
        PV_Table_MV = pd.DataFrame()
        PL_Table_MV = pd.DataFrame()
        V_Table_MV = pd.DataFrame()
        Flow_Table_MV = pd.DataFrame()
        PBLV_Table_MV = pd.DataFrame()
        PSC_Table_MV = pd.DataFrame()
        PSE_Table_MV = pd.DataFrame()
        Lmabda2_Table_MV = pd.DataFrame()

        r=0
        for (bd,i) in mv.Omega_S:
            for t in mv.T_L:
                PSMV_Table_LV.loc[r,'Bd'] = bd
                PSMV_Table_LV.loc[r,'Bus'] = i
                PSMV_Table_LV.loc[r,t] = round(mv.Lps[bd,i,t].value,self.Result_Decimals)*mv.sb
                PBE_Table_LV.loc[r,'Bd'] = bd
                PBE_Table_LV.loc[r,'Bus'] = i
                PBE_Table_LV.loc[r,t] = round(mv.Lpbe[bd,i,t].value,self.Result_Decimals)*mv.sb

                PBC_Table_LV.loc[r,'Bd'] = bd
                PBC_Table_LV.loc[r,'Bus'] = i
                PBC_Table_LV.loc[r,t] = round(mv.Lpbc[bd,i,t].value,self.Result_Decimals)*mv.sb
            r += 1

        r=0
        for (bd,i) in mv.A:
            for t in mv.T_L:
                PV_Table_LV.loc[r,'Bd'] = bd
                PV_Table_LV.loc[r,'Bus'] = i
                PV_Table_LV.loc[r,t] = round(mv.pv[bd,i,t].value,self.Result_Decimals)*mv.sb
                PL_Table_LV.loc[r,'Bd'] = bd
                PL_Table_LV.loc[r,'Bus'] = i
                PL_Table_LV.loc[r,t] = mv.PL_L[bd,i,t]
                SOC_Table_LV.loc[r,'Bd'] = bd
                SOC_Table_LV.loc[r,'Bus'] = i
                SOC_Table_LV.loc[r,t] = round(mv.soc[bd,i,t].value,self.Result_Decimals)*mv.sb
                CH_Table_LV.loc[r,'Bd'] = bd
                CH_Table_LV.loc[r,'Bus'] = i
                CH_Table_LV.loc[r,t] = round(mv.ch[bd,i,t].value,self.Result_Decimals)*mv.sb
                DS_Table_LV.loc[r,'Bd'] = bd
                DS_Table_LV.loc[r,'Bus'] = i
                DS_Table_LV.loc[r,t] = round(mv.ds[bd,i,t].value,self.Result_Decimals)*mv.sb
                W_Table_LV.loc[r,'Bd'] = bd
                W_Table_LV.loc[r,'Bus'] = i
                W_Table_LV.loc[r,t] = round(mv.w[bd,i,t].value,self.Result_Decimals)
            r += 1

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
        for i in mv.A_M:
            for t in mv.T_M:
                PG_Table_MV.loc[r,'bus'] = i
                PG_Table_MV.loc[r,t] = round(mv.pg_M[i,t].value,self.Result_Decimals)*self.sb_MV
                PV_Table_MV.loc[r,'bus'] = i
                PV_Table_MV.loc[r,t] = round(mv.pv_M[i,t].value,self.Result_Decimals)*self.sb_MV
                PL_Table_MV.loc[r,'bus'] = i
                PL_Table_MV.loc[r,t] = mv.PL_M[i,t]
            r += 1


        r=0
        for (i,j) in mv.L_M:
            for t in mv.T_M:
                Flow_Table_MV.loc[r,'from'] = str(i)
                Flow_Table_MV.loc[r,'to'] = str(j)
                Flow_Table_MV.loc[r,t] = round(mv.P_M[i,j,t].value,self.Flow_Result_Decimals)*self.sb_MV
            r += 1  

        with pd.ExcelWriter(self.Results_KKT_File) as data:
            PBLV_Table_MV.to_excel(data, sheet_name="PBLV")
            PSE_Table_MV.to_excel(data, sheet_name="PSE") 
            PSC_Table_MV.to_excel(data, sheet_name="PSC") 
            PG_Table_MV.to_excel(data, sheet_name="PG")
            PV_Table_MV.to_excel(data, sheet_name="PV")
            PL_Table_MV.to_excel(data, sheet_name="PL")
            V_Table_MV.to_excel(data, sheet_name="V")
            Flow_Table_MV.to_excel(data, sheet_name="Flow")
            PSMV_Table_LV.to_excel(data, sheet_name="PSMV_LV")
            PBE_Table_LV.to_excel(data, sheet_name="PBE_LV")
            PBC_Table_LV.to_excel(data, sheet_name="PBC_LV")
            PV_Table_LV.to_excel(data, sheet_name="PV_LV")
            PL_Table_LV.to_excel(data, sheet_name="PL_LV")
            SOC_Table_LV.to_excel(data, sheet_name="SOC_LV")
            CH_Table_LV.to_excel(data, sheet_name="CH_LV")
            DS_Table_LV.to_excel(data, sheet_name="DS_LV")
            W_Table_LV.to_excel(data, sheet_name="W_LV")
            Lmabda2_Table_MV.to_excel(data, sheet_name="Lambda2")

    def Print_Results_KKT_MV_Only(self,mv):
        PV_Table_LV = pd.DataFrame()
        PL_Table_LV = pd.DataFrame()
        SOC_Table_LV = pd.DataFrame()
        CH_Table_LV = pd.DataFrame()
        DS_Table_LV = pd.DataFrame()
        W_Table_LV = pd.DataFrame()
        PSMV_Table_LV = pd.DataFrame()
        PBE_Table_LV = pd.DataFrame()
        PBC_Table_LV = pd.DataFrame()

        PG_Table_MV = pd.DataFrame()
        PV_Table_MV = pd.DataFrame()
        PL_Table_MV = pd.DataFrame()
        V_Table_MV = pd.DataFrame()
        Flow_Table_MV = pd.DataFrame()
        PBLV_Table_MV = pd.DataFrame()
        PSC_Table_MV = pd.DataFrame()
        PSE_Table_MV = pd.DataFrame()
        Lmabda2_Table_MV = pd.DataFrame()


        r=0
        for i in mv.Omega_M:
            for t in mv.T_M:
                V_Table_MV.loc[r,'bus'] = i
                V_Table_MV.loc[r,t] = round(mv.V[i,t].value,self.Result_Decimals)
            r += 1
        
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
                Flow_Table_MV.loc[r,t] = round(np.sqrt((mv.P[i,j,t].value)**2 + (mv.Q[i,j,t].value)**2)/ (mv.s_max[i,j] / self.sb_MV),self.Flow_Result_Decimals)
            r += 1  

        with pd.ExcelWriter(self.Results_MV_Only_File) as data:
            PBLV_Table_MV.to_excel(data, sheet_name="PBLV")
            PSE_Table_MV.to_excel(data, sheet_name="PSE") 
            PSC_Table_MV.to_excel(data, sheet_name="PSC") 
            PG_Table_MV.to_excel(data, sheet_name="PG")
            PV_Table_MV.to_excel(data, sheet_name="PV")
            PL_Table_MV.to_excel(data, sheet_name="PL")
            V_Table_MV.to_excel(data, sheet_name="V")
            Flow_Table_MV.to_excel(data, sheet_name="Flow")
            PSMV_Table_LV.to_excel(data, sheet_name="PSMV_LV")
            PBE_Table_LV.to_excel(data, sheet_name="PBE_LV")
            PBC_Table_LV.to_excel(data, sheet_name="PBC_LV")
            PV_Table_LV.to_excel(data, sheet_name="PV_LV")
            PL_Table_LV.to_excel(data, sheet_name="PL_LV")
            SOC_Table_LV.to_excel(data, sheet_name="SOC_LV")
            CH_Table_LV.to_excel(data, sheet_name="CH_LV")
            DS_Table_LV.to_excel(data, sheet_name="DS_LV")
            W_Table_LV.to_excel(data, sheet_name="W_LV")
            Lmabda2_Table_MV.to_excel(data, sheet_name="Lambda2")

if __name__ == "__main__":
    runing = Binivel_MV_LV_Problem()
    runing.ReadExcelFile_LV(runing.File_LV)
    runing.ReadExcelFile_MV(runing.File_MV)
    sm = runing.Algorithm()
    runing.Print_Results_KKT(sm)