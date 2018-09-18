import pandas as pd
import numpy as np
import pyodbc
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, normalize
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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
				,DP.WellFlac
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
	df.loc[:, 'DateKey'] = pd.to_datetime(df.loc[:, 'DateKey'])

	df.loc[:, 'rolling'] = df.loc[:, 'LinePressure'].rolling(4, center=False).mean()
	df.loc[:, 'rolling_std'] = df.loc[:, 'LinePressure'].rolling(4, center=False).std()
	df.loc[:, 'rolling_gas'] = df.loc[:, 'AllocatedGas'].rolling(4, center=False).mean()

	X = df['rolling_std'].values
	X_1 = df['rolling'].values
	X_gas = df['rolling_gas'].values

	y_pred = np.ones(X.shape)
	y_pred[X > 35] = 0

	pressure_vals = df.loc[:, 'LinePressure'].values.reshape(-1, 1)
	X_pred = linear(df.loc[:, 'DateKey'].astype('int64').values.reshape(-1, 1),
					pressure_vals)
	# std = np.std(df.loc[:, 'LinePressure'].values)
	# upper = (X_pred + (1.96 * std)).reshape(-1, 1)
	# lower = (X_pred - (1.96 * std)).reshape(-1, 1)

	std = np.std(df.loc[df['LinePressure'] > 0, 'LinePressure'].values)
	med = df.loc[df['LinePressure'] > 0, 'LinePressure'].median()
	upper = med + std * .5
	lower = med - std * .5

	df['spike'] = np.where((df['LinePressure'] > upper) |
						   (df['LinePressure'] < lower), 1, 0)
	df['3_spike'] = np.where((((df['spike'] == df['spike'].shift(1)) &
							   (df['spike'] == df['spike'].shift(2))) |
							  ((df['spike'] == df['spike'].shift(-1)) &
						  	   (df['spike'] == df['spike'].shift(-2))) |
							  ((df['spike'] == df['spike'].shift(1)) &
						  	   (df['spike'] == df['spike'].shift(-1)))) &
							  df['spike'] == 1, 1, 0)

	df['5_spike'] = np.where((((df['spike'] == df['spike'].shift(-4)) &
							   (df['spike'] == df['spike'].shift(-3)) &
							   (df['spike'] == df['spike'].shift(-2)) &
							   (df['spike'] == df['spike'].shift(-1))) |
							  ((df['spike'] == df['spike'].shift(-3)) &
  							   (df['spike'] == df['spike'].shift(-2)) &
  							   (df['spike'] == df['spike'].shift(-1)) &
  							   (df['spike'] == df['spike'].shift(1))) |
							  ((df['spike'] == df['spike'].shift(-2)) &
  							   (df['spike'] == df['spike'].shift(-1)) &
  							   (df['spike'] == df['spike'].shift(1)) &
  							   (df['spike'] == df['spike'].shift(2))) |
							  ((df['spike'] == df['spike'].shift(-1)) &
  							   (df['spike'] == df['spike'].shift(1)) &
  							   (df['spike'] == df['spike'].shift(2)) &
  							   (df['spike'] == df['spike'].shift(3))) |
							  ((df['spike'] == df['spike'].shift(1)) &
  							   (df['spike'] == df['spike'].shift(2)) &
  							   (df['spike'] == df['spike'].shift(3)) &
  							   (df['spike'] == df['spike'].shift(4)))) &
							  df['spike'] == 1, 1, 0)

	out_pressure = (pressure_vals > upper).astype(int) + (pressure_vals < lower).astype(int)

	df.loc[:, 'press_norm'] = normalize(df['LinePressure'].values.reshape(-1, 1))
	df.loc[:, 'gas_norm'] = normalize(df['AllocatedGas'].values.reshape(-1, 1))

	# plot_linear(df, X_pred, upper, lower, df['API'].unique()[0])
	# plot_it(df, X, X_1, X_gas, y_pred, df['API'].unique()[0])

	# rf_regressor(df.loc[df['5_spike'] == 1, :])

	return df

# June 11th - 22nd
# Normalize each well, then run regression on the whole set

def rf_regressor(df):
	X = df['press_norm'].values.reshape(-1, 1)
	# X_norm = normalize(X)
	y = df['gas_norm'].values.reshape(-1, 1)
	# y_norm = normalize(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y,
														test_size=0.2,
														random_state=13)

	rf = RandomForestRegressor()

	rf.fit(X_train, y_train)
	print('Score')
	print(rf.score(X_test, y_test))

def linear(X, y):
	scaler = StandardScaler().fit(X, y)

	X_scale = scaler.transform(X)

	poly = PolynomialFeatures(degree=2)
	X_ = poly.fit_transform(X)

	# model = make_pipeline(PolynomialFeatures(3), LinearRegression())
	# model.fit(X, y)

	lr = LinearRegression()
	lr.fit(X_, y)

	return lr.predict(X_).flatten()

def plot_linear(df, line, upper, lower, api):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	std = np.std(df.loc[df['LinePressure'] > 0, 'LinePressure'].values)
	med = df.loc[df['LinePressure'] > 0, 'LinePressure'].median()
	upper = med + std * .5
	lower = med - std * .5

	ax.plot(df['DateKey'], df['LinePressure'], color='black', linestyle='--', alpha=.3)
	ax.plot(df.loc[(df['LinePressure'] <= upper) & (df['LinePressure'] >= lower), 'DateKey'],
			df.loc[(df['LinePressure'] <= upper) & (df['LinePressure'] >= lower),'LinePressure'],
			color='black')
	ax.plot(df['DateKey'], df['AllocatedGas'], color='#4286f4', linestyle='--', alpha=.3)
	ax.plot(df.loc[(df['LinePressure'] <= upper) & (df['LinePressure'] >= lower), 'DateKey'],
			df.loc[(df['LinePressure'] <= upper) & (df['LinePressure'] >= lower),'AllocatedGas'],
			color='#42d9f4')

	# df.loc[:, 'flag_up'] = (df.loc[:, 'LinePressure'] > upper).astype(int)
	# df.loc[:, 'flag_down'] = (df.loc[:, 'LinePressure'] < lower).astype(int)
	# df.loc[:, 'flag'] = df.loc[:, ['flag_up', 'flag_down']].sum(axis=1)
	#
	# for i in range(0, len(df['DateKey'].values)):
	# 	ax.axvspan(i, i+1, color='red', alpha=df.loc[:, 'flag'].values[i]/4)

	ax.axhline(upper, color='b', linestyle='--')
	ax.axhline(lower, color='b', linestyle='--')
	ax.axhline(med, color='r')

	plt.savefig('figures/{}_linear.png'.format(api))

def plot_it(df, X, X_1, X_gas, y_pred, api):
	plt.close()

	fig, ax = plt.subplots()
	ax.plot(X_1, label='Line Pressure')
	ax.plot(X_gas, label='Gas', color='black')
	for i in range(0, len(X)):
		ax.axvspan(i, i+1, color='red', alpha=((y_pred[i]-1)*-1)/4)
	# ax.plot(flagged * 100, 'xb')
	plt.ylabel('Line Pressure (PSI)')
	plt.xlabel('Day')
	plt.legend()
	plt.title('{} 14 day rolling'.format(api))
	plt.savefig('figures/{}_14rolling.png'.format(api))

def neural_net(df):
	df.sort_values('DateKey', inplace=True)
	df.loc[:, 'line_sq'] = df.loc[:, 'LinePressure'] ** 2
	df.loc[:, 'line_cu'] = df.loc[:, 'LinePressure'] ** 3
	df.loc[:, 'rolling'] = df.loc[:, 'LinePressure'].rolling(4, center=False).mean()
	df.loc[:, 'rolling_std'] = df.loc[:, 'LinePressure'].rolling(4, center=False).std()

	df = delta(df)
	df.fillna(0, inplace=True)
	# df.loc[:, 'line_change'] = df.loc[:, 'LinePressure'] - df.loc[:, 'LinePressure'].shift(1)
	# df.loc[:, 'line_change'].fillna(0, inplace=True)

	y = df['AllocatedGas']
	X = df[['LinePressure', 'rolling', 'rolling_std', 'line_sq', 'line_cu',
			'line_change']]

	mlp = MLPRegressor()
	mlp.fit(X, y)

	print('Score: {}'.format(mlp.score(X, y)))

def delta(df):
	return_df = pd.DataFrame(columns=df.columns)

	for api in df['API'].unique():
		well_df = df.loc[df['API'] == api]
		well_df.loc[:, 'line_change'] = well_df.loc[:, 'LinePressure'] - \
										well_df.loc[:, 'LinePressure'].shift(1)
		well_df.loc[:, 'line_change'].fillna(0, inplace=True)
		return_df = return_df.append(well_df)

	return return_df

if __name__ == '__main__':
	# df = data_pull()
	# df.to_csv('data/line_pressure.csv')
	df = pd.read_csv('data/line_pressure.csv')

	flacs = pd.read_csv('data/kelvin_wellflacs.csv', header=None).values.flatten()
	kelvin_df = df.loc[df['WellFlac'].isin(flacs), :]

	# neural_net(df)

	apis = sorted(kelvin_df['API'].unique())
	spike_df = pd.DataFrame()

	for api in apis:
		api_df = anomaly(kelvin_df.loc[kelvin_df['API'] == api,
									  ['API', 'DateKey', 'LinePressure', 'AllocatedGas']])
		spike_df = spike_df.append(api_df)

	rf_regressor(spike_df.loc[spike_df['3_spike'] == 1, :])
