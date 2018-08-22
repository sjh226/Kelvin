import pandas as pd
import numpy as np
import pyodbc
import sys


def well_pull():
	try:
		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
									r'Server=10.75.6.160, 1433;'
									r'Database=OperationsDataMart;'
									r'trusted_connection=yes'
									)
	except pyodbc.Error:
		print("Connection Error")
		sys.exit()

	cursor = connection.cursor()
	SQLCommand = ("""
		DROP TABLE IF EXISTS #Normal
		DROP TABLE IF EXISTS #Plunger

		SELECT	P.Wellkey
				,DP.WellFlac
				,W.API
				,DP.BusinessUnit
				,DP.DateKey
				,DP.AllocatedOil
				,DP.AllocatedGas
				,DP.AllocatedWater
				,P.LastChokeEntry
				,P.LastChokeStatus
			INTO #Normal
			FROM [Business].[Operations].[DailyProduction] DP
			JOIN [OperationsDataMart].[Dimensions].[Wells] W
			  ON W.WellFlac = DP.WellFlac
			JOIN [OperationsDataMart].[Facts].[Production] P
			  ON P.Wellkey = W.Wellkey
			  AND P.DateKey = DP.DateKey
			WHERE P.LastChokeStatus = 'Normal Operations'
			  AND DP.DateKey >= DATEADD(year, -2, GETDATE())
			  --AND DP.AllocatedGas > 0
			  --AND DP.AllocatedOil > 0
			  --AND DP.AllocatedWater > 0
	""")

	cursor.execute(SQLCommand)

	SQLCommand = ("""
		SELECT DISTINCT N.API
						,N.Wellkey
						,N.WellFlac
						,N.BusinessUnit
						,CASE WHEN AvGas.AvgGas IS NOT NULL
							  THEN AvGas.AvgGas
							  ELSE 0 END AS AvgGas
						,CASE WHEN N.BusinessUnit = 'North'
								THEN (LG.Weighted_OilGasRatio + LG.Weighted_WaterGasRatio)
								ELSE ((CASE WHEN AvOil.AvgOil IS NOT NULL
											  THEN AvOil.AvgOil
											  ELSE 0 END + CASE WHEN AvWat.AvgWater IS NOT NULL
																  THEN AvWat.AvgWater
																  ELSE 0 END) / CASE WHEN AvGas.AvgGas IS NOT NULL
																				  THEN AvGas.AvgGas
																				  ELSE NULL END) END AS LGR
						,CASE WHEN PT.plungerType LIKE '%Stock%' OR PT.plungerType LIKE '%stock%' OR
									PT.plungerType LIKE '%Cleanout%' OR PT.plungerType LIKE '%Snake%' OR
									PT.plungerType LIKE '%Venturi%' OR PT.plungerType LIKE '%Viper%' OR PT.plungerType LIKE '%Vortex%'
								THEN 'Bar Stock'
								WHEN PT.plungerType LIKE '%acemaker%' OR PT.plungerType LIKE '%Center%' OR
									PT.plungerType LIKE '%ypass%' OR PT.plungerType LIKE '%Sleeve%' OR
									PT.plungerType LIKE '%y-pass%'
								THEN 'Pacemaker/Bypass'
								WHEN PT.plungerType LIKE '%Other%' OR PT.plungerType LIKE '%6%' OR
									PT.plungerType LIKE '%8%' OR PT.plungerType LIKE '%Brush%' OR
									PT.plungerType LIKE '%Sphere%'
								THEN 'Other'
								WHEN PT.plungerType IS NULL
								THEN NULL
								ELSE 'Padded' END AS PlungerType
			--INTO #Plunger
			FROM (SELECT	API
						,Wellkey
						,WellFlac
						,BusinessUnit
						--,(AVG(AllocatedOil) + AVG(AllocatedWater)) / AVG(AllocatedGas) AS LGR
					FROM #Normal
					GROUP BY API, Wellkey, BusinessUnit, WellFlac) N
			LEFT OUTER JOIN (SELECT  API
						,AVG(AllocatedGas) AS AvgGas
					FROM #Normal
					WHERE AllocatedGas > 0
					GROUP BY API) AvGas
			  ON AvGas.API = N.API
			LEFT OUTER JOIN (SELECT	API
									,AVG(AllocatedOil) AS AvgOil
							FROM #Normal
							WHERE AllocatedOil > 0
							GROUP BY API) AvOil
			  ON AvOil.API = N.API
			LEFT OUTER JOIN (SELECT  API
						,AVG(AllocatedWater) AS AvgWater
				FROM #Normal
				WHERE AllocatedWater > 0
				GROUP BY API) AvWat
			  ON AvWat.API = N.API
			LEFT OUTER JOIN [TeamOptimizationEngineering].[dbo].[NorthLGR4] LG
			ON LG.WellKey = N.Wellkey
			LEFT OUTER JOIN (SELECT	A.apiNumber
									,P.plungerManufacturer
									,P.plungerType
								FROM [EDW].[Enbase].[PlungerInspection] P
								JOIN [EDW].[Enbase].[Asset] A
								  ON A._id = P.assetId
								INNER JOIN (SELECT	A.apiNumber
													,MAX(PlI.createdDate) AS MaxDate
											FROM [EDW].[Enbase].[PlungerInspection] PlI
											JOIN [EDW].[Enbase].[Asset] A
											  ON A._id = PlI.assetId
											GROUP BY A.apiNumber) MaxP
								  ON MaxP.apiNumber = A.apiNumber
								  AND MaxP.MaxDate = P.createdDate
								WHERE A.assetType = 'Well') PT
			ON LEFT(PT.apiNumber, 10) = N.API
			--WHERE PT.plungerType IS NOT NULL
			WHERE N.Wellkey IN (SELECT	MAX(DISTINCT(Wellkey))
								FROM #Normal
								GROUP BY API)
	""")

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
		df.drop_duplicates(inplace=True)
	except:
		df = None
		print('Dataframe is empty')

	df.loc[:,'API'] = df.loc[:,'API'].astype(float)

	return df

def kelvin_api(val):
	if len(str(val)) < 10:
		return int('0' + str(val))
	else:
		return val

def cluster(df):
	north_df = df.loc[df['BusinessUnit'] == 'North', :]
	west_df = df.loc[df['BusinessUnit'] == 'West', :]
	midcon_df = df.loc[df['BusinessUnit'] == 'Midcon', :]

	north_lgr = [sorted(north_df['LGR'].values)[round(len(north_df['LGR'].values)/3)],
				 sorted(north_df['LGR'].values)[round((len(north_df['LGR'].values)/3) * 2)]]

	def north_lgr_logic(row):
		if row['LGR'] < north_lgr.min():
			return 1
		elif row['LGR'] >= north_lgr.max():
			return 3
		else:
			return 2

if __name__ == '__main__':
	all_df = well_pull()
	kelvin_df = pd.read_csv('data/l48_plungers.csv')
	kelvin_df['API'] = kelvin_df['API'].apply(kelvin_api)

	k_df = pd.merge(kelvin_df, all_df, on='API', how='left')
	k_df.loc[:,'WellFlac'] = pd.to_numeric(k_df.loc[:,'WellFlac'], errors='coerce')

	compressor_df = pd.read_csv('data/west_compressors.csv', encoding='ISO-8859-1')
	comp_df = compressor_df[[' WellFlac ', ' Compressor Manufacturer ']]
	comp_df.rename(columns={' Compressor Manufacturer ': 'Comp',
							' WellFlac ': 'WellFlac'}, inplace=True)

	comp_df.loc[:,'Comp'] = np.where(comp_df.loc[:,'Comp'].notnull(), 1, 0)

	df = pd.merge(k_df, comp_df, on='WellFlac', how='left')
	df.drop_duplicates(inplace=True)
	df = df.loc[(df['BusinessUnit'] == 'West') |
				((df['BusinessUnit'] != 'West') & (df['Comp'].isnull())), :]

	cluster(df)
