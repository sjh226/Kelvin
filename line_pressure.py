import pandas as pd
import numpy as np
import pyodbc
import sys


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


if __name__ == '__main__':
	df = data_pull()
