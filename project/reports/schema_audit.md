# HAI 23.05 schema audit

- reference file: `hai-train1.csv` (87 columns)
- hai-train1.csv: 87 columns
- hai-train2.csv: 87 columns
- hai-train3.csv: 87 columns
- hai-train4.csv: 87 columns
- hai-test1.csv: 87 columns
- hai-test2.csv: 87 columns

- tag count (excluding `timestamp`): 86

## Tag summary
| tag | process | kind | discrete | unit | observed range |
|---|---|---|---|---|---|
| P1_FCV01D | P1_boiler | command | False | unknown | [0, 80.34] |
| P1_FCV01Z | P1_boiler | feedback | False | unknown | [0.3143, 80.87] |
| P1_FCV02D | P1_boiler | command | False | unknown | [15, 100] |
| P1_FCV02Z | P1_boiler | feedback | False | unknown | [12.29, 97.6] |
| P1_FCV03D | P1_boiler | command | False | unknown | [42.45, 71.54] |
| P1_FCV03Z | P1_boiler | feedback | False | unknown | [43.26, 72.73] |
| P1_FT01 | P1_boiler | measurement | False | unknown | [0.801, 447.3] |
| P1_FT01Z | P1_boiler | feedback | False | unknown | [4.171, 1339] |
| P1_FT02 | P1_boiler | measurement | False | unknown | [1.183, 2506] |
| P1_FT02Z | P1_boiler | feedback | False | unknown | [6.157, 3190] |
| P1_FT03 | P1_boiler | measurement | False | unknown | [167.9, 287.5] |
| P1_FT03Z | P1_boiler | feedback | False | unknown | [821.8, 1072] |
| P1_LCV01D | P1_boiler | command | False | unknown | [0, 71.36] |
| P1_LCV01Z | P1_boiler | feedback | False | unknown | [0.2991, 71.24] |
| P1_LIT01 | P1_boiler | measurement | False | unknown | [171.4, 461.4] |
| P1_PCV01D | P1_boiler | command | False | unknown | [0.8701, 56.63] |
| P1_PCV01Z | P1_boiler | feedback | False | unknown | [0.3525, 57.21] |
| P1_PCV02D | P1_boiler | command | False | unknown | [12, 24.43] |
| P1_PCV02Z | P1_boiler | feedback | False | unknown | [11.67, 24.63] |
| P1_PIT01 | P1_boiler | measurement | False | unknown | [0.04593, 2.049] |
| P1_PIT01_HH | P1_boiler | measurement | False | unknown | [5.243e+05, 5.243e+05] |
| P1_PIT02 | P1_boiler | measurement | False | unknown | [0.3122, 1.543] |
| P1_PP01AD | P1_boiler | status | True | unknown | [0, 0] |
| P1_PP01AR | P1_boiler | status | True | unknown | [0, 0] |
| P1_PP01BD | P1_boiler | status | True | unknown | [1, 1] |
| P1_PP01BR | P1_boiler | status | True | unknown | [1, 1] |
| P1_PP02D | P1_boiler | command | True | unknown | [1, 1] |
| P1_PP02R | P1_boiler | status | True | unknown | [1, 1] |
| P1_PP04 | P1_boiler | measurement | False | unknown | [0, 45] |
| P1_PP04D | P1_boiler | command | True | unknown | [0, 1] |
| P1_PP04SP | P1_boiler | setpoint | False | unknown | [27.13, 27.98] |
| P1_SOL01D | P1_boiler | command | False | unknown | [5.243e+05, 5.243e+05] |
| P1_SOL03D | P1_boiler | command | False | unknown | [5.243e+05, 5.243e+05] |
| P1_STSP | P1_boiler | setpoint | True | unknown | [1, 1] |
| P1_TIT01 | P1_boiler | measurement | False | unknown | [27.85, 34.03] |
| P1_TIT02 | P1_boiler | measurement | False | unknown | [32.49, 39.6] |
| P1_TIT03 | P1_boiler | measurement | False | unknown | [25.61, 30.53] |
| P2_24Vdc | P2_turbine | measurement | False | unknown | [28.01, 28.05] |
| P2_ATSW_Lamp | P2_turbine | status | True | unknown | [0, 1] |
| P2_AutoGO | P2_turbine | status | True | unknown | [0, 1] |
| P2_AutoSD | P2_turbine | command | False | unknown | [101.1, 129.7] |
| P2_Emerg | P2_turbine | status | True | unknown | [0, 0] |
| P2_MASW | P2_turbine | status | True | unknown | [0, 1] |
| P2_MASW_Lamp | P2_turbine | status | True | unknown | [0, 1] |
| P2_ManualGO | P2_turbine | status | True | unknown | [0, 1] |
| P2_ManualSD | P2_turbine | command | False | unknown | [0, 12.93] |
| P2_OnOff | P2_turbine | status | True | unknown | [1, 1] |
| P2_RTR | P2_turbine | measurement | False | unknown | [2880, 2880] |
| P2_SCO | P2_turbine | measurement | False | unknown | [5.415e+04, 5.484e+04] |
| P2_SCST | P2_turbine | measurement | False | unknown | [719.4, 721.2] |
| P2_SIT01 | P2_turbine | measurement | False | rpm | [746, 967] |
| P2_TripEx | P2_turbine | status | True | unknown | [1, 1] |
| P2_VIBTR01 | P2_turbine | measurement | False | um | [5.123, 7.617] |
| P2_VIBTR02 | P2_turbine | measurement | False | um | [-4.074, -1.848] |
| P2_VIBTR03 | P2_turbine | measurement | False | um | [7.989, 10.51] |
| P2_VIBTR04 | P2_turbine | measurement | False | um | [-1.882, 0.3069] |
| P2_VT01 | P2_turbine | measurement | False | unknown | [11.79, 12.05] |
| P2_VTR01 | P2_turbine | measurement | False | unknown | [50, 50] |
| P2_VTR02 | P2_turbine | measurement | False | unknown | [50, 50] |
| P2_VTR03 | P2_turbine | measurement | False | unknown | [50, 50] |
| P2_VTR04 | P2_turbine | measurement | False | unknown | [50, 50] |
| P3_FIT01 | P3_water | measurement | False | unknown | [20, 5450] |
| P3_LCP01D | P3_water | command | False | unknown | [-8, 1.382e+04] |
| P3_LCV01D | P3_water | command | False | unknown | [-176, 1.79e+04] |
| P3_LH01 | P3_water | measurement | False | unknown | [70, 70] |
| P3_LIT01 | P3_water | measurement | False | unknown | [8015, 1.973e+04] |
| P3_LL01 | P3_water | measurement | False | unknown | [30, 30] |
| P3_PIT01 | P3_water | measurement | False | unknown | [16, 4724] |
| P4_HT_FD | P4_hil | measurement | False | pu | [-0.02792, 0.03067] |
| P4_HT_PO | P4_hil | measurement | False | MW | [0, 83.6] |
| P4_HT_PS | P4_hil | measurement | False | MW | [0.07477, 7.94] |
| P4_LD | P4_hil | measurement | False | MW | [233.3, 499] |
| P4_ST_FD | P4_hil | measurement | False | pu | [-0.00955, 0.00883] |
| P4_ST_GOV | P4_hil | measurement | False | unknown | [1.304e+04, 2.628e+04] |
| P4_ST_LD | P4_hil | measurement | False | MW | [232, 491.9] |
| P4_ST_PO | P4_hil | measurement | False | MW | [237.3, 482.6] |
| P4_ST_PS | P4_hil | measurement | False | MW | [1.133, 28.92] |
| P4_ST_PT01 | P4_hil | measurement | False | bar | [9669, 1.023e+04] |
| P4_ST_TT01 | P4_hil | measurement | False | degC | [2.686e+04, 2.747e+04] |
| x1001_05_SETPOINT_OUT | DCS_internal | controller_output | False | unknown | [0.05148, 0.1976] |
| x1001_15_ASSIGN_OUT | DCS_internal | controller_output | False | unknown | [0.7783, 1.595] |
| x1002_07_SETPOINT_OUT | DCS_internal | controller_output | False | unknown | [302.6, 398.6] |
| x1002_08_SETPOINT_OUT | DCS_internal | controller_output | False | unknown | [950.6, 1042] |
| x1003_10_SETPOINT_OUT | DCS_internal | controller_output | False | unknown | [6.157, 3190] |
| x1003_18_SETPOINT_OUT | DCS_internal | controller_output | False | unknown | [25.19, 28] |
| x1003_24_SUM_OUT | DCS_internal | controller_output | False | unknown | [28.07, 33.09] |
