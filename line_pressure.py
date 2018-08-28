import pandas as pd
import numpy as np
import pyodbc
import sys
import matplotlib.pyplot as plt


def normal_pull():
	# Trusted connection allows for Windows authentication
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

		SELECT	P.Wellkey
				,W.API
				,P.DateKey
				,P.LinePressure
				,P.TubingPressure
				,P.CasingPressure
				,P.MeasuredOil
				,P.MeasuredGas
				,P.MeasuredWater
				,P.LastChokeEntry
				,P.LastChokeStatus
				,ROW_NUMBER() OVER (PARTITION BY P.Wellkey ORDER BY P.DateKey DESC) AS RowNum
		  INTO #Normal
		  FROM [OperationsDataMart].[Facts].[Production] P
		  JOIN [OperationsDataMart].[Dimensions].[Wells] W
			ON W.Wellkey = P.Wellkey
		  WHERE P.LinePressure IS NOT NULL
			AND W.BusinessUnit = 'North'
			AND P.LastChokeStatus = 'Normal Operations'
	""")

	cursor.execute(SQLCommand)

	SQLCommand = ("""
		SELECT  API
				,AVG(LinePressure) AS AvgLine
				,STDEV(LinePressure) AS Deviation
				,AVG(TubingPressure) AS AvgTubing
				,AVG(CasingPressure) AS AvgCasing
		  FROM #Normal
		  WHERE RowNum <= 30
		  GROUP BY API
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

	return df

def data_pull():
	# Trusted connection allows for Windows authentication
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
		SELECT	P.Wellkey
				,DP.API10 AS API
				,DP.DateKey
				,P.LinePressure
				,DP.AllocatedOil
				,DP.AllocatedGas
				,DP.AllocatedWater
				,P.LastChokeEntry
				,P.LastChokeStatus
		  FROM [Business].[Operations].[DailyProduction] DP
		  JOIN [OperationsDataMart].[Dimensions].[Wells] W
			ON W.WellFlac = DP.WellFlac
		  JOIN [OperationsDataMart].[Facts].[Production] P
		    ON P.Wellkey = W.Wellkey
			AND P.DateKey = DP.DateKey
		  WHERE P.LinePressure IS NOT NULL
			AND W.BusinessUnit = 'North'
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

	return df

def anomaly(df):
	df.sort_values('DateKey', inplace=True)

	df.loc[:, 'rolling'] = df.loc[:, 'LinePressure'].rolling(21, center=False).mean()
	df.loc[:, 'rolling_std'] = df.loc[:, 'LinePressure'].rolling(21, center=False).std()
	df.loc[:, 'rolling_gas'] = df.loc[:, 'AllocatedGas'].rolling(21, center=False).mean()

	X = df['rolling_std'].values
	X_1 = df['rolling'].values
	X_gas = df['rolling_gas'].values

	print(X)

	y_pred = np.ones(X.shape)
	y_pred[X > 15] = 0

	plot_it(X, X_1, X_gas, y_pred, df['API'].unique()[0])

def plot_it(X, X_1, X_gas, y_pred, api):
	plt.close()

	fig, ax = plt.subplots()
	ax.plot(X_1, label='Line Pressure')
	ax.plot(X_gas, label='Gas', color='black')
	for i in range(0, len(X)):
		ax.axvspan(i, i+1, color='red', alpha=((y_pred[i]-1)*-1)/4)
	plt.ylabel('Line Pressure (PSI)')
	plt.xlabel('Day')
	plt.legend()
	plt.title('{} 14 day rolling'.format(api))
	plt.savefig('figures/{}_14rolling_10.png'.format(api))


if __name__ == '__main__':
	# df = data_pull()
	# df.to_csv('data/line_pressure.csv')
	df = pd.read_csv('data/line_pressure.csv')

	apis = sorted(df['API'].unique())

	for api in apis[:3]:
		anomaly(df.loc[df['API'] == api, ['API', 'DateKey', 'LinePressure',
										  'AllocatedGas']])
